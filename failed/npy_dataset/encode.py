import os
import numpy as np
from tqdm import tqdm
from pipeline.datasets.sevir.sevir import SEVIRLightningDataModule
from omegaconf import OmegaConf
import shutil
import pandas as pd
import glob

EXPORT_BASE = "/home/vatsal/Dataserver/NWM/datasets/preprocessed_vil"
os.makedirs(EXPORT_BASE, exist_ok=True)

def remove_preprocessed_data():
    if os.path.exists(EXPORT_BASE):
        shutil.rmtree(EXPORT_BASE)

SPLITS = ["val", "train", "test"]
for split in SPLITS:
    os.makedirs(os.path.join(EXPORT_BASE, split), exist_ok=True)

def init_dm():
    cfg = OmegaConf.load("pipeline/datasets/sevir/config.yaml").Dataset
    dm = SEVIRLightningDataModule(cfg=cfg)
    dm.prepare_data()
    dm.setup()#train 731374

def process_split(split: str, EXPORT_BASE: str, dm):
    loader = getattr(dm, f"{split}_dataloader")()
    out_dir = os.path.join(EXPORT_BASE, split)

    for batch in tqdm(loader, desc=f"Encoding {split}", unit="batch"):
        vil, event_idx, seq_idx = batch  # vil: (B, T, H, W, C)
        b, t, h, w, c = vil.shape
        assert t == 49, f"Expected 49 frames, got {t}"

        assert len(event_idx) == len(seq_idx) == b, f"Expected {b} event_idx and seq_idx, got {(event_idx)} and {(seq_idx)}"
        # Extract single frame
        assert (seq_idx == 0).all(), f"Expected seq_idx to be all zeros, got {seq_idx}"

        frame = vil.squeeze(-1).float()  # (b, 49, 384, 384)
        assert frame.shape == (b, 49, 384, 384), f"Unexpected shape: {frame.shape}"

        for batch_idx in range(b):
            for seq_idx in range(t):
                path = os.path.join(out_dir, f"{split}_{event_idx[batch_idx]:08d}_{seq_idx:08d}.npy")
                np.save(path, frame[batch_idx, seq_idx].numpy())

# for s in SPLITS:
#     process_split(s)

# print("All splits encoded.")

def sort_key(file_path):
    fn = os.path.basename(file_path)
    event_idx = int(fn.split("_")[-2])
    seq_idx   = int(fn.split("_")[-1].split(".")[0])
    return (event_idx, seq_idx)
def save_catalog(split: str):
    files = sorted(glob.glob(os.path.join(EXPORT_BASE, split, "*.npy")))
    sorted_files = sorted(files, key=sort_key)
    df = pd.DataFrame(sorted_files, columns = ["file_path"])
    df.to_csv(os.path.join(EXPORT_BASE, f"{split}_catalog.csv"), index = False)
    print(f"Saved catalog for {split}")

# for s in SPLITS:
#     save_catalog(s)

"""
Done on 
Dataset:
  name: sevir
  data_dir: "/home/vatsal/Dataserver/NWM/datasets/sevir"
  seq_len: 49
  sample_mode: "sequent"
  stride: 1
  layout: "NTHWC"
  rescale_method: "01"
  preprocess: true
  verbose: false
  aug_mode: "0"
  ret_contiguous: true
  batch_size: 16
  num_workers: 16
  seed: 0
  val_ratio: 0.0
  start_data: null
  train_test_split_date: 2019-06-01
  end_data: null

in 7hrs,933 train, 254 test
train points = 731472
"""