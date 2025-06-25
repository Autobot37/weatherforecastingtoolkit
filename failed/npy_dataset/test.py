from pipeline.datasets.sevir.sevir import SEVIRLightningDataModule
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import random
import torch
import os
import pandas as pd

"""
failed didnt worked since bottleneck is in I/O confirmed by scalene profiler.
"""

class FastDataset(Dataset):
    def __init__(self, data_dir, seq_len=49, stride=12, split="train", sanity_check=False):
        self.data_dir = data_dir
        self.split = split
        self.seq_len = seq_len
        self.stride = stride
        catalog_path = os.path.join(data_dir, f"{split}_catalog.csv")
        self.catalog = pd.read_csv(catalog_path)
        
        self.file_paths = self.catalog["file_path"].values
        
        self.frames_per_event = 49
        self._num_seq_per_event = (self.frames_per_event - self.seq_len) // self.stride + 1
        self._total_events = len(self.catalog) // self.frames_per_event

        if sanity_check:
            self._sanity_check(random.choice(self.catalog["file_path"].tolist()))

    def sort_key(self, file_path):
        fn = os.path.basename(file_path)
        event_idx = int(fn.split("_")[-2])
        seq_idx   = int(fn.split("_")[-1].split(".")[0])
        return (event_idx, seq_idx)

    def _sanity_check(self, file_path):
        fn = os.path.basename(file_path)
        event_idx, _ = self.sort_key(fn)
        count = sum(1 for fp in self.catalog["file_path"].tolist() if self.sort_key(fp)[0] == event_idx)
        assert count == self.frames_per_event, (
            f"Event {event_idx} has {count} files, expected {self.frames_per_event}"
        )

    def num_seq_per_event(self):
        return self._num_seq_per_event

    def get_total_events(self):
        return self._total_events

    def __len__(self):
        return self.get_total_events() * self.num_seq_per_event()

    def __getitem__(self, idx):
        event_idx = idx // self._num_seq_per_event
        window_idx = idx % self._num_seq_per_event

        start = event_idx * self.frames_per_event + window_idx * self.stride
        end = start + self.seq_len

        window_files = self.file_paths[start:end]

        arrays = [np.load(fp) for fp in window_files]
        return np.stack(arrays, axis=0).astype(np.float32)

def main():
    cfg = OmegaConf.load("pipeline/datasets/sevir/config.yaml").Dataset
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    seq_len = cfg.seq_len
    stride = cfg.stride
    fast_dataset = FastDataset(cfg.encoded_data_dir, seq_len=seq_len, stride=stride, split="test", sanity_check=False)
    fast_dataloader = DataLoader(fast_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, prefetch_factor=64, pin_memory=True)

    data_module = SEVIRLightningDataModule(cfg=cfg)
    data_module.setup()
    test_loader = data_module.test_dataloader()

    print("performing sanity check")
    idx = 0
    for batch1, batch2 in zip(test_loader, fast_dataloader):
        print(batch1.shape, batch2.shape)
        assert torch.allclose(batch1.squeeze(-1), batch2)
        idx += 1
        if idx > 5:
            break
    print("sanity check passed")

    import time
    from tqdm import tqdm

    print("profiling for slow dataset")
    start_time = time.time()
    idx = 0
    total = len(test_loader)
    for batch in tqdm(test_loader, total=total):
        idx += 1
        if idx > 100:
            break
    end_time = time.time()
    print(f"Time taken for slow dataset: {end_time - start_time} seconds")

    print("profiling for fast dataset")
    start_time = time.time()
    idx = 0
    for batch in tqdm(fast_dataloader, total=len(fast_dataset)):
        idx += 1
        if idx > 10:
            break
    end_time = time.time()
    print(f"Time taken for fast dataset: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()