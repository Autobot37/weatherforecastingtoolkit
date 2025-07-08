import os
import argparse
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import OmegaConf
from termcolor import colored
from pipeline.models.autoencoderkl.losses.contperceptual import adopt_weight, hinge_d_loss, NLayerDiscriminator, weights_init
from pipeline.datasets.sevir.sevir import SEVIRLightningDataModule
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pipeline.helpers import load_checkpoint_cascast, log_gradients_paramater, modelcheckpointcallback, TrackGradNormCallback \
    , adamw_optimizer, cosine_warmup_scheduler, log_metrics, log_wandb_images
"""
384x384
rec = l1
L_adv = -mean(D(x^))
w_adapt = grad(rec)/grad(l_adv) #to balance both losses
gen_loss = rec + disc_factor * w_adapt * L_adv
"""
os.environ['WANDB_API_KEY'] = '041eda3850f131617ee1d1c9714e6230c6ac4772'

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = nn.GELU()
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
            
        self.act2 = nn.GELU()

    def forward(self, x):
        shortcut = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += shortcut
        return self.act2(out)

class UpsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.resblock = ResidualBlock(in_ch, out_ch, stride=1)

    def forward(self, x):
        return self.resblock(self.upsample(x))

class ConvAutoencoder(nn.Module):
    def __init__(self, in_ch=1, latent_dim=512):
        super().__init__()
        
        # Encoder: 128 → 64 → 32 → 16 → 4 → 1
        self.enc1 = ResidualBlock(in_ch,   64, stride=2)  # 128 → 64
        self.enc2 = ResidualBlock(64,     128, stride=2)  # 64 → 32
        self.enc3 = ResidualBlock(128,    256, stride=2)  # 32 → 16
        self.enc4 = ResidualBlock(256,    512, stride=4)  # 16 → 4
        self.enc5 = ResidualBlock(512,   1024, stride=4)  # 4 → 1

        self.flatten = nn.Flatten()                     # (B,1024,1,1) → (B,1024)
        self.fc_enc = nn.Linear(1024, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, 1024)
        self.unflatten = nn.Unflatten(1, (1024, 1, 1))   # (B,1024) → (B,1024,1,1)
        self.dec_init_conv = ResidualBlock(1024, 1024, stride=1)

        # Decoder: 1 → 4 → 16 → 32 → 64 → 128
        self.dec1 = UpsampleBlock(1024, 512, scale_factor=4)  # 1 → 4
        self.dec2 = UpsampleBlock(512,  256, scale_factor=4)  # 4 → 16
        self.dec3 = UpsampleBlock(256,  128, scale_factor=2)  # 16 → 32
        self.dec4 = UpsampleBlock(128,   64, scale_factor=2)  # 32 → 64
        
        self.final_upsample = nn.Upsample(scale_factor=2, mode='nearest') # 64 → 128
        self.final_conv = nn.Conv2d(64, in_ch, kernel_size=3, stride=1, padding=1)

    def encode(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        x = self.flatten(x)
        z = self.fc_enc(x)
        return z

    def decode(self, z):
        x = self.fc_dec(z)
        x = self.unflatten(x)
        x = self.dec_init_conv(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.final_upsample(x)
        x = self.final_conv(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, z

class Loss(nn.Module):
    def __init__(self, disc_start, disc_num_layers=3, disc_in_channels=1, disc_weight=1.0, use_actnorm=False):
        super().__init__()
        self.disc_start = disc_start
        self.disc_weight = disc_weight

        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm
        ).apply(weights_init)

    def calculate_adaptive_weight(self, rec_loss, disc_loss, last_layer):
        rec_grad = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
        disc_grad = torch.autograd.grad(disc_loss, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(rec_grad) / (torch.norm(disc_grad) + 1e-4)
        d_weight = self.disc_weight * d_weight
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight

    def forward(self, inputs, reconstructions, optimizer_idx, last_layer, split, global_step):
        rec_loss = F.l1_loss(reconstructions, inputs, reduction="mean")

        if global_step < self.disc_start:
            return rec_loss, {
            f"{split}/total_loss": rec_loss.detach().cpu(),
            f"{split}/rec_loss": rec_loss.detach().cpu(),
            f"{split}/g_loss": torch.tensor(0.0).cpu(),
            f"{split}/d_weight": torch.tensor(0.0).cpu(),
            }

        if optimizer_idx == 0:
            logits_fake = self.discriminator(reconstructions)
            g_loss = -torch.mean(logits_fake)
            d_weight = self.calculate_adaptive_weight(rec_loss, g_loss, last_layer)
            loss = rec_loss + d_weight * g_loss
            return loss, {
            f"{split}/total_loss": loss.detach().cpu(),
            f"{split}/rec_loss": rec_loss.detach().cpu(),
            f"{split}/g_loss": g_loss.detach().cpu(),
            f"{split}/d_weight": d_weight.detach().cpu(),
            }

        if optimizer_idx == 1:
            logits_real = self.discriminator(inputs.detach())
            logits_fake = self.discriminator(reconstructions.detach())
            d_loss = hinge_d_loss(logits_real, logits_fake)
            return d_loss, {
            f"{split}/disc_loss": d_loss.detach().cpu(),
            f"{split}/logits_real": logits_real.detach().mean().cpu(),
            f"{split}/logits_fake": logits_fake.detach().mean().cpu(),
            }


class Model(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.autoencoder = ConvAutoencoder()
        self.loss = Loss(cfg.lpips.disc_start, 
                        disc_num_layers=cfg.lpips.disc_num_layers, 
                        disc_in_channels=cfg.lpips.disc_in_channels, 
                        disc_weight=cfg.lpips.disc_weight, 
                        use_actnorm=cfg.lpips.use_actnorm)
        self.input_frames =  cfg.dataset.input_frames
        self.pred_frames = cfg.dataset.pred_frames
        self.total_steps = cfg.trainer.total_train_steps
        self.accumulate_grad_batches = cfg.trainer.accumulate_grad_batches

        self.automatic_optimization = False

    def forward(self, x):
        out, posterior = self.autoencoder(x)
        return out
    
    def get_last_layer(self):
        return self.autoencoder.final_conv.weight

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()
        g_sch, d_sch = self.lr_schedulers()       

        inp = batch.permute(0,3,1,2) #[b, c, h, w]
        pred = self(inp)

        aeloss, log_dict_ae = self.loss(inp, pred, optimizer_idx = 0, last_layer = self.get_last_layer(), split="train", global_step=self.global_step)
        self.log_dict(log_dict_ae, on_step=True, on_epoch = True, sync_dist=True)
        aeloss = aeloss / self.accumulate_grad_batches

        self.manual_backward(aeloss)
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            self.clip_gradients(g_opt, gradient_clip_val=self.cfg.optim.gradient_clip_val)
            g_opt.step()
            g_sch.step()
            g_opt.zero_grad(set_to_none=True)

        #discriminator step
        discloss, log_dict_disc = self.loss(inp, pred, optimizer_idx = 1, last_layer = self.get_last_layer(), split="train", global_step=self.global_step)
        self.log_dict(log_dict_disc, on_step=True, on_epoch = True, sync_dist=True)
        discloss = discloss / self.accumulate_grad_batches

        self.manual_backward(discloss)
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            self.clip_gradients(d_opt, gradient_clip_val=self.cfg.optim.gradient_clip_val)
            d_opt.step()
            d_sch.step()
            d_opt.zero_grad(set_to_none=True)
        
        log_interval = int(self.cfg.logging.log_train_all_metrics_n * self.cfg.trainer.total_train_steps)
        if batch_idx % log_interval == 0:
            log_metrics(pred.unsqueeze(2), inp.unsqueeze(2), "train", self)

        plot_interval = int(self.cfg.logging.log_train_plots_n * self.cfg.trainer.total_train_steps)
        if batch_idx % plot_interval == 0:
            log_wandb_images(pred, inp, f"Train Reconstruction vs Original_epoch_{self.current_epoch}_batch_{batch_idx}", self)
    
    def validation_step(self, batch, batch_idx):
        inp = batch.permute(0,3,1,2) #[b, c, h, w]
        pred = self(inp)

        aeloss, log_dict_ae = self.loss(inp, pred, optimizer_idx = 0, last_layer = self.get_last_layer(), split="val", global_step=self.global_step)
        self.log_dict(log_dict_ae, on_step=True, on_epoch = True, sync_dist=True)
        discloss, log_dict_disc = self.loss(inp, pred, optimizer_idx = 1, last_layer = self.get_last_layer(), split="val", global_step=self.global_step)
        self.log_dict(log_dict_disc, on_step=True, on_epoch = True, sync_dist=True)
        
        log_metrics(pred.unsqueeze(2), inp.unsqueeze(2), "val", self)

        plot_interval = int(self.cfg.logging.log_val_plots_n * self.cfg.trainer.total_val_steps)
        if batch_idx % plot_interval == 0:
            log_wandb_images(pred, inp, f"Val_Reconstruction vs Original_epoch_{self.current_epoch}_batch_{batch_idx}", self)

    def test_step(self, batch, batch_idx):
        inp = batch.permute(0,3,1,2) #[b, c, h, w]
        pred = self(inp)
        
        aeloss, log_dict_ae = self.loss(inp, pred, optimizer_idx = 0, last_layer = self.get_last_layer(), split="test", global_step=self.global_step)
        self.log_dict(log_dict_ae, on_step=True, on_epoch = True, sync_dist=True)
        discloss, log_dict_disc = self.loss(inp, pred, optimizer_idx = 1, last_layer = self.get_last_layer(), split="test", global_step=self.global_step)
        self.log_dict(log_dict_disc, on_step=True, on_epoch = True, sync_dist=True)
        
        log_metrics(pred.unsqueeze(2), inp.unsqueeze(2), "test", self)

        plot_interval = int(self.cfg.logging.log_val_plots_n * self.cfg.trainer.total_val_steps)
        if batch_idx % plot_interval == 0:
            log_wandb_images(pred, inp, f"Test_Reconstruction vs Original_epoch_{self.current_epoch}_batch_{batch_idx}_test", self)
        return aeloss
        
    def configure_optimizers(self):
        opt_ae = adamw_optimizer(self.autoencoder, self.cfg.optim.lr, self.cfg.optim.weight_decay)
        sch_params = self.cfg.cosine_warmup
        warmup_steps = sch_params.warmup_ratio * self.total_steps
        sch_ae = cosine_warmup_scheduler(opt_ae, sch_params.start_lr, sch_params.final_lr, sch_params.peak_lr, self.total_steps, warmup_steps)

        opt_disc = adamw_optimizer(self.autoencoder, self.cfg.optim.lr, self.cfg.optim.weight_decay)
        sch_params = self.cfg.cosine_warmup
        warmup_steps = sch_params.warmup_ratio * self.total_steps
        sch_disc = cosine_warmup_scheduler(opt_disc, sch_params.start_lr, sch_params.final_lr, sch_params.peak_lr, self.total_steps, warmup_steps)
        return [
                {"optimizer": opt_ae, "lr_scheduler": {"scheduler": sch_ae, "interval": "step", "frequency": 1}},
                {"optimizer": opt_disc, "lr_scheduler": {"scheduler": sch_disc, "interval": "step", "frequency": 1}}
            ]
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    args = parser.parse_args()
    resume_ckpt = args.resume
    run_id = args.run_id
    if run_id is not None:
        print(colored(f"Resuming from checkpoint: {resume_ckpt} with run_id: {run_id}", "green"))

    torch.backends.cudnn.benchmark = True 
    torch.set_float32_matmul_precision('high')

    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg = OmegaConf.load(config_path)
    outputs_path = os.path.join(cfg.experiment_path, 'outputs')
    os.makedirs(outputs_path, exist_ok=True)

    dm = SEVIRLightningDataModule(cfg.dataset); dm.setup()
    for loader in [dm.train_dataloader(), dm.val_dataloader()]:
        for data in loader:
            print(f"Data shape: {data.shape}")
            break

    total_train_steps = (len(dm.train_dataloader()) * cfg.trainer.max_epochs) / cfg.trainer.accumulate_grad_batches
    total_val_steps = (len(dm.val_dataloader()) * cfg.trainer.max_epochs) / cfg.trainer.accumulate_grad_batches
    total_test_steps = (len(dm.test_dataloader()) * cfg.trainer.max_epochs) / cfg.trainer.accumulate_grad_batches
    cfg.trainer.total_train_steps = int(total_train_steps)
    cfg.trainer.total_val_steps = int(total_val_steps)
    cfg.trainer.total_test_steps = int(total_test_steps)
    if cfg.trainer.limit_train_batches is not None:
        cfg.trainer.total_train_steps = total_train_steps * cfg.trainer.limit_train_batches
    if cfg.trainer.limit_val_batches is not None:
        cfg.trainer.total_val_steps = total_val_steps * cfg.trainer.limit_val_batches
    if cfg.trainer.limit_test_batches is not None:
        cfg.trainer.total_test_steps = total_test_steps * cfg.trainer.limit_test_batches
    cfg.lpips.disc_start = int(cfg.lpips.disc_start * cfg.trainer.total_train_steps)

    logger = WandbLogger(project = cfg.project_name, name = cfg.experiment_name, save_dir = os.path.join(cfg.experiment_path, 'outputs'), id = run_id, resume = "allow")
    run_id = logger.experiment.id
    run_dir = logger.experiment.dir
    artifact = wandb.Artifact(cfg.experiment_name, type="code")
    artifact.add_file(os.path.join(os.path.dirname(__file__), "train.py"))
    logger.experiment.log_artifact(artifact)

    checkpoint_callback = modelcheckpointcallback(run_dir, cfg.trainer.total_train_steps, cfg.trainer.save_every_n_steps, cfg.trainer.save_on_train_epoch_end)
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs, 
        accelerator='cpu', 
        # devices=cfg.trainer.devices,
        callbacks=[checkpoint_callback, lr_monitor_callback, TrackGradNormCallback()],
        logger=logger,
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        limit_test_batches=cfg.trainer.limit_test_batches,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
    )

    model = Model(cfg)
    trainer.fit(model, dm, ckpt_path=resume_ckpt)
    print("done")