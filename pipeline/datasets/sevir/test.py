from pipeline.datasets.sevir.sevir import SEVIRLightningDataModule
from omegaconf import OmegaConf

import time
from tqdm import tqdm

def main():
    cfg = OmegaConf.load("pipeline/datasets/sevir/fast_config.yaml").Dataset

    data_module = SEVIRLightningDataModule(cfg=cfg)
    data_module.setup()
    train_loader = data_module.train_dataloader()
    
    print("profiling for dataset")
    start_time = time.time()
    idx = 0
    total = len(train_loader)
    for batch in tqdm(train_loader, total=total):
        idx += 1
        if idx > 1000:
            break
    end_time = time.time()
    print(f"Time taken for dataset: {end_time - start_time} seconds")

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