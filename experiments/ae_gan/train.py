import os
import argparse
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import OmegaConf
from termcolor import colored
from pipeline.models.autoencoderkl.losses.contperceptual import LPIPSWithDiscriminator
from pipeline.datasets.sevir.sevir import SEVIRLightningDataModule
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pipeline.helpers import load_checkpoint_cascast, log_gradients_paramater, modelcheckpointcallback, TrackGradNormCallback \
    , adamw_optimizer, cosine_warmup_scheduler, log_metrics, log_wandb_images
"""
384x384
"""

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = nn.GELU()
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        # Skip connection path
        # If stride is not 1 or input/output channels are different, we need to project the skip connection
        # to match the output dimensions of the main path.
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
        # Using 'nearest' mode for upsampling avoids checkerboard artifacts that can be
        # caused by transposed convolutions.
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        # After upsampling, we use a residual block to learn and refine features.
        self.resblock = ResidualBlock(in_ch, out_ch, stride=1)

    def forward(self, x):
        return self.resblock(self.upsample(x))

class ConvAutoencoder(nn.Module):
    def __init__(self, in_ch=1, latent_dim=512):
        super().__init__()
        
        # Encoder: 384 → 96 → 48 → 24 → 6 → 1
        self.enc1 = ResidualBlock(in_ch,  64, stride=4)  # 384→96
        self.enc2 = ResidualBlock(64,    128, stride=2)  # 96→48
        self.enc3 = ResidualBlock(128,   256, stride=2)  # 48→24
        self.enc4 = ResidualBlock(256,   512, stride=4)  # 24→6
        self.enc5 = ResidualBlock(512,  1024, stride=6)  # 6→1

        self.flatten = nn.Flatten()             # (B,1024,1,1) → (B,1024)
        self.fc_enc  = nn.Linear(1024, latent_dim)

        self.fc_dec  = nn.Linear(latent_dim, 1024)
        self.unflatten = nn.Unflatten(1, (1024, 1, 1)) # (B,1024) → (B,1024,1,1)
        
        self.dec_init_conv = ResidualBlock(1024, 1024, stride=1)

        # Decoder: 1 → 6 → 24 → 48 → 96 → 384
        self.dec1 = UpsampleBlock(1024, 512, scale_factor=6)  # 1→6
        self.dec2 = UpsampleBlock(512,  256, scale_factor=4)  # 6→24
        self.dec3 = UpsampleBlock(256,  128, scale_factor=2)  # 24→48
        self.dec4 = UpsampleBlock(128,   64, scale_factor=2)  # 48→96
        
        self.final_upsample = nn.Upsample(scale_factor=4, mode='nearest') # 96→384
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

class Model(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.autoencoder = ConvAutoencoder()
        self.loss = LPIPSWithDiscriminator(cfg.lpips)
        self.input_frames =  cfg.dataset.input_frames
        self.pred_frames = cfg.dataset.pred_frames
        self.total_steps = cfg.trainer.total_train_steps

        self.automatic_optimization = False

    def forward(self, x):
        out, posterior = self.autoencoder(x)
        return out, posterior
    
    def get_last_layer(self):
        return self.autoencoder.final_conv.weight

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()
        g_sch, d_sch = self.lr_schedulers()       

        inp = batch.permute(0,3,1,2) #[b, c, h, w]
        pred, posterior = self.autoencoder(inp)

        aeloss, log_dict_ae = self.loss()

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        log_interval = int(self.cfg.logging.log_train_all_metrics_n * self.cfg.trainer.total_train_steps)
        if batch_idx % log_interval == 0:
            pred = pred + inp_t
            tgt = tgt + inp_t

            decoded_pred = self.autoencoder.decode(pred)
            decoded_tgt = self.autoencoder.decode(tgt)
            log_metrics(decoded_pred, decoded_tgt, "train", self)

        plot_interval = int(self.cfg.logging.log_train_plots_n * self.cfg.trainer.total_train_steps)
        if batch_idx % plot_interval == 0:
            log_wandb_images(decoded_pred, decoded_tgt, f"Reconstruction vs Original_epoch_{self.current_epoch}_batch_{batch_idx}", self)
        return loss
    
    def validation_step(self, batch, batch_idx):
        v = batch.permute(0,3,1,2).unsqueeze(2)
        v = self.autoencoder.encode(v)
        b, t, c, h, w = v.shape
        inp, tgt = v[:, :self.input_frames], v[:, self.input_frames:]
        inp_t = inp[:, -1].unsqueeze(1)
        inp = inp - inp_t
        tgt = tgt - inp_t
        pred = self(inp.reshape(b, self.input_frames * c,  h * w))
        pred = pred.reshape(b, self.pred_frames, c, h, w)
        loss = F.mse_loss(pred, tgt)
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        
        pred = pred + inp_t
        tgt = tgt + inp_t

        decoded_pred = self.autoencoder.decode(pred)
        decoded_tgt = self.autoencoder.decode(tgt)

        log_metrics(decoded_pred, decoded_tgt, "val", self)

        plot_interval = int(self.cfg.logging.log_val_plots_n * self.cfg.trainer.total_val_steps)
        if batch_idx % plot_interval == 0:
            log_wandb_images(decoded_pred, decoded_tgt, f"Reconstruction vs Original_epoch_{self.current_epoch}_batch_{batch_idx}", self)
        return loss

    def test_step(self, batch, batch_idx):
        v = batch.permute(0,3,1,2).unsqueeze(2)
        v = self.autoencoder.encode(v)
        b, t, c, h, w = v.shape
        inp, tgt = v[:, :self.input_frames], v[:, self.input_frames:]
        inp_t = inp[:, -1].unsqueeze(1)
        inp = inp - inp_t
        tgt = tgt - inp_t
        pred = self(inp.reshape(b, self.input_frames, c * h * w)).reshape(b, self.pred_frames, c, h, w)
        loss = F.mse_loss(pred, tgt)
        self.log('test_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        
        pred = pred + inp_t
        tgt = tgt + inp_t

        decoded_pred = self.autoencoder.decode(pred)
        decoded_tgt = self.autoencoder.decode(tgt)

        log_metrics(decoded_pred, decoded_tgt, "test", self)

        plot_interval = int(self.cfg.logging.log_val_plots_n * self.cfg.trainer.total_val_steps)
        if batch_idx % plot_interval == 0:
            log_wandb_images(decoded_pred, decoded_tgt, f"Reconstruction vs Original_epoch_{self.current_epoch}_batch_{batch_idx}_test", self)
        return loss
        
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
                {"optimizer": opt_ae, "lr_scheduler": sch_ae},
                {"optimizer": opt_disc, "lr_scheduler": sch_disc},
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
        accelerator='gpu', 
        devices=cfg.trainer.devices,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback, lr_monitor_callback],
        logger=logger,
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        limit_test_batches=cfg.trainer.limit_test_batches,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
    )

    model = Model(cfg)
    trainer.fit(model, dm, ckpt_path=resume_ckpt)
    print("done")