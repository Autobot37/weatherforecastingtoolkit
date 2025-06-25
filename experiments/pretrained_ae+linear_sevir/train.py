import os
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import OmegaConf
from termcolor import colored
from pipeline.models.autoencoderkl.autoencoder_kl import AutoencoderKL
from pipeline.utils import load_checkpoint_cascast
from pipeline.datasets.sevir.sevir import SEVIRLightningDataModule
from pytorch_lightning.callbacks import LearningRateMonitor
from pipeline.basemodel import BaseModel
"""
384x384
"""
os.environ['WANDB_API_KEY'] = '041eda3850f131617ee1d1c9714e6230c6ac4772'

class Autoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.autoencoder = AutoencoderKL(**config)
        self.autoencoder.eval() 
        self.scaling_factor = 0.18125
        load_checkpoint_cascast("/home/vatsal/NWM/PreDiff/scripts/vae/sevirlr/autoencoder_ckpt.pth", self.autoencoder)
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.autoencoder.requires_grad_(False)

    @torch.no_grad()
    def encode(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        out = []
        for i in range(T):
            frame = x[:, i]  # (B, C, H, W)
            z = self.autoencoder.encode(frame).sample()
            out.append(z.unsqueeze(1))
        return torch.cat(out, dim=1)
        # out = self.autoencoder.encode(x.reshape(B*T, C, H, W)).mode()
        # return out.reshape(B, T, 4, 48, 48)

    @torch.no_grad()
    def decode(self, x):
        # x: (B, T, latent_C, H, W)
        B, T, C, H, W = x.shape
        out = []
        for i in range(T):
            frame = x[:, i]
            dec = self.autoencoder.decode(frame)
            out.append(dec.unsqueeze(1))
        return torch.cat(out, dim=1)
        # out = self.autoencoder.decode(x.reshape(B*T, C, H, W))
        # return out.reshape(B, T, 1, 384, 384)

class Model(BaseModel):
    def __init__(self, cfg):
        super().__init__()
        cfg = OmegaConf.load("config.yaml")
        self.cfg = cfg
        self.autoencoder = Autoencoder(cfg.Autoencoder)
        self.input_frames =  cfg.input_frames
        self.pred_frames = cfg.pred_frames
        self.total_steps = cfg.total_steps
        self.predictor = nn.Linear(self.input_frames * 4, self.pred_frames * 4)

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(colored(f"Total trainable parameters: {total_params:,}", 'blue'))
        print(colored(f"Input frames: {self.input_frames}, Predicted frames: {self.pred_frames}", 'yellow'))

    def forward(self, x):
        out = self.predictor(x)
        return out

    def training_step(self, batch, batch_idx):
        v = batch['vil'].permute(0,3,1,2).unsqueeze(2)
        v = self.autoencoder.encode(v)  # (B, T, LC, LH, LW)
        b, t, c, h, w = v.shape
        inp, tgt = v[:, :self.input_frames], v[:, self.input_frames:]
        inp_t = inp[:, -1].unsqueeze(1)
        inp = inp - inp_t
        tgt = tgt - inp_t

        pred = self(inp.permute(0, 3, 4, 1, 2).reshape(b, h, w, 13*c)).permute(0, 3, 1, 2).reshape(b, 12, c, h, w)
        loss = F.mse_loss(pred, tgt)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        v = batch['vil'].permute(0,3,1,2).unsqueeze(2)
        v = self.autoencoder.encode(v)
        b, t, c, h, w = v.shape
        inp, tgt = v[:, :self.input_frames], v[:, self.input_frames:]
        inp_t = inp[:, -1].unsqueeze(1)
        inp = inp - inp_t
        tgt = tgt - inp_t
        pred = self(inp.permute(0, 3, 4, 1, 2).reshape(b, h, w, 13*c)).permute(0, 3, 1, 2).reshape(b, 12, c, h, w)
        loss = F.mse_loss(pred, tgt)
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        
        pred = pred + inp_t
        tgt = tgt + inp_t

        decoded_pred = self.autoencoder.decode(pred)
        decoded_tgt = self.autoencoder.decode(tgt)

        if batch_idx % self.cfg.logging.log_val_all_metrics_n == 0:
            self.log_metrics(decoded_pred, decoded_tgt, "val", self.logger)

        if batch_idx % self.cfg.logging.log_val_plots_n == 0:
            data = {"data" : {"target" : decoded_tgt.unsqueeze(2), "pred" : decoded_pred.unsqueeze(2)}, "label" : "reconstruction vs original", "name" : f"val_{self.current_epoch}_{batch_idx}"}
            self.log_wandb_images(data, plot_fn, self.logger)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True 
    torch.set_float32_matmul_precision('high')

    cfg = OmegaConf.load("config.yaml")
    name = cfg.logging.wandb_name
    outputs_path = os.path.join(cfg.experiment_path, 'outputs')
    os.makedirs(outputs_path, exist_ok=True)

    checkpoint_callback = BaseModel.modelcheckpointcallback(cfg)
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')

    logger = BaseModel.get_logger(cfg)
    art = wandb.Artifact(name, type="code")
    art.add_file(os.path.join(os.path.dirname(__file__), "pretrained_ae_linear.py"))
    logger.experiment.log_artifact(art)

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs, 
        accelerator='gpu', 
        devices=cfg.trainer.devices,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback, lr_monitor_callback],
        logger=logger,
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        track_grad_norm = 2,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        log_every_n_steps=1,
    )

    dm = SEVIRLightningDataModule(cfg.Dataset); dm.setup()
    for loader in [dm.train_dataloader(), dm.val_dataloader()]:
        for sample in loader:
            data = sample["vil"]
            print(f"Data shape: {data.shape}")
            break
    plot_fn = SEVIRLightningDataModule.plot_sample

    model = Model(cfg)
    trainer.fit(model, dm)