import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import WandbLogger
from termcolor import colored
from pipeline.metrics import calc_metrics
import os
from pytorch_lightning.callbacks import ModelCheckpoint

class BaseModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def adamw_optimizer(self, model, cfg):
        """
        https://towardsdatascience.com/weight-decay-and-its-peculiar-effects-66e0aee3e7b8/
        weight_decay in model architecture testing should be 0 since 
        we are testing the mdoels ability to learn.
        otherwise generally it should be 1e-2 for 50m models.
        it is a regularizater and need tuning also helps in finding simple intepretation if overparametrized.
        if using another regularization like dropout and data augmentations then still needed 
        since dropout forces more robustness and data augmentations forces invariance.
        """
        return torch.optim.AdamW(model.parameters(), lr = cfg.lr, weight_decay = cfg.weight_decay)

    def cosine_warmup_scheduler(self, opt, cfg):
        """
        start->base_lr[in warmup]
        base_lr->final_lr[in rest steps] in single 2nd half of cosine cycle.
        general_values for scratch
        warmup_steps generally 10%.
        peak_lr generally 1e-3
        final_lr generally 1e-6
        start_lr generally 1e-4
        """
        start_lr = cfg.cosine_warmup.start_lr
        final_lr = cfg.cosine_warmup.final_lr
        peak_lr = cfg.cosine_warmup.peak_lr
        total_steps = cfg.trainer.total_step
        warmup_steps = cfg.cosine_warmup.warmup_ratio * total_steps

        for param_group in opt.param_groups:
            if param_group['lr'] != peak_lr:
                print(colored(f"lr of {param_group['name']} is not peak lr, it is {param_group['lr']} changing to {peak_lr}", 'red'))
                param_group['lr'] = peak_lr

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            opt,
            start_factor = start_lr / peak_lr,
            end_factor = 1.0,
            total_iters = warmup_steps
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max = total_steps - warmup_steps,
            eta_min = final_lr
        )
        sch = torch.optim.lr_scheduler.SequentialLR(
            opt,
            schedulers = [warmup_scheduler, cosine_scheduler],
            milestones = [warmup_steps]
        )
        return sch
    
    def one_cycle_scheduler(self, opt, cfg):
        """
        start_lr -> peak_lr[in warmup steps]
        peak_lr -> final_lr[in rest steps]
        general_values
        peak_lr generally 1e-3
        start_lr generally peak_lr / 25 ~ 4e-5
        final_lr generally start_lr / 100 ~ 4e-7
        rampup_steps = 30% of total steps

        rampup steps should be higher like 1/4 of total steps.
        """
        start_lr = cfg.one_cycle.start_lr
        peak_lr = cfg.one_cycle.peak_lr
        final_lr = cfg.one_cycle.final_lr
        total_steps = cfg.trainer.total_step
        rampup_steps = cfg.one_cycle.rampup_ratio * total_steps

        if rampup_steps / total_steps < 0.2:
            print(colored(f"rampup steps should be higher than 20% of total steps, it is {rampup_steps / total_steps}", 'red'))

        # rampup percentage
        pct_start = rampup_steps / total_steps

        div_factor = peak_lr / start_lr
        final_div_factor = start_lr / final_lr

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=peak_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            anneal_strategy='cos'
        )

        return scheduler

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        super().lr_scheduler_step(scheduler, optimizer_idx, metric)

    @staticmethod
    def get_logger(cfg, run_id = None):
        return WandbLogger(project = cfg.project_name, name = cfg.experiment_name, save_dir = os.path.join(cfg.experiment_path, 'outputs'), id = run_id)

    @staticmethod
    def log_metrics(predictions, targets, tag, logger):
        """
        predictions: (B, T, C, H, W) [0, 1]
        targets: (B, T, C, H, W) [0, 1]
        """
        metrics = calc_metrics(predictions, targets)
        metrics = {f"{tag}_{k}": v for k, v in metrics.items()}
        logger.log_dict(metrics, on_step=True, on_epoch=True, sync_dist=True)

    @staticmethod
    def log_wandb_images(data, plot_fn, logger):
        fig = plot_fn(data)
        if isinstance(logger, WandbLogger):
            logger.log_image(key = data["name"], images = [fig])
            plt.close(fig)
        else:
            raise ValueError("logger is not a WandbLogger")
    
    @staticmethod
    def log_gradients_paramater(model, cfg, logger):
        """
        one time call for logging gradients and parameters and model architecture
        """
        if isinstance(logger, WandbLogger):
            logger.watch(model, log = "all", log_freq = cfg.logging.wandb_watch_log_freq)

    @staticmethod
    def modelcheckpointcallback(cfg):
        return ModelCheckpoint(
            dirpath = os.path.join(cfg.experiment_path, 'outputs', 'checkpoints'),
            filename = "{epoch}-{step:06d}",
            save_top_k = 1,
            every_n_train_steps = cfg.trainer.save_every_n_steps,
            save_on_train_epoch_end = cfg.trainer.save_on_train_epoch_end
        )
        