from pipeline.datasets.sevir.sevir import SEVIRLightningDataModule
from omegaconf import OmegaConf

import time
from tqdm import tqdm
import faulthandler
faulthandler.enable()

def main():
    # cfg = OmegaConf.load("pipeline/datasets/sevir/fast_config.yaml").dataset

    # data_module = SEVIRLightningDataModule(cfg=cfg)
    # data_module.setup()
    # train_loader = data_module.train_dataloader()
    
    # start_time = time.time()
    # idx = 0
    # total = len(train_loader)
    # for epochs in range(50):
    #     for batch in tqdm(train_loader, total=total):
    #         pass

    from pipeline.datasets.sevire.sevir import SEVIRLightningDataModule
    data = SEVIRLightningDataModule(dataset_name="sevir_lr", num_workers=8, batch_size=8)
    data.setup()
    data.prepare_data()

    train_loader = data.train_dataloader()
    start_time = time.time()
    for epoch in range(10):
        for batch in tqdm(train_loader):
            pass
    
    end_time = time.time()
    print(f"Time taken for 10 epochs: {end_time - start_time} seconds")


if __name__ == "__main__":
    main()

"""
so system is doing 61% means i/0 bottleneck
and native doing 15%
python doing 18%
so not really that can be dont better than that
tested on config
Dataset:
  name: sevir
  data_dir: "/home/vatsal/Dataserver/NWM/datasets/sevir"
  encoded_data_dir: "/home/vatsal/Dataserver/NWM/datasets/preprocessed_vil"
  seq_len: 25
  sample_mode: "sequent"
  stride: 12
  layout: "NHWT"
  rescale_method: "01"
  preprocess: true
  verbose: false
  aug_mode: "0"
  ret_contiguous: true
  batch_size: 8
  num_workers: 8
  seed: 0
  val_ratio: 0.0
  start_data: null
  train_test_split_date: 2019-06-01
  end_data: null

on test_loader
1000 batches tool 10 minutes.
"""

"""

"""