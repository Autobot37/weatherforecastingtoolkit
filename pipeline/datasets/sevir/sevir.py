"""Code is adapted from https://github.com/amazon-science/earth-forecasting-transformer/blob/e60ff41c7ad806277edc2a14a7a9f45585997bd7/src/earthformer/datasets/sevir/sevir_dataloader.py."""
import os
from typing import List, Union, Dict, Sequence, Optional, Tuple
from math import ceil
import numpy as np
import numpy.random as nprand
import datetime
import pandas as pd
import h5py
import torch
from torch.nn.functional import avg_pool2d
from einops import rearrange
from omegaconf import OmegaConf, DictConfig
import math
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset as TorchDataset, DataLoader, random_split
from torchvision import transforms
from pytorch_lightning import LightningDataModule, seed_everything
from copy import deepcopy
from matplotlib.colors import BoundaryNorm
import torchvision.transforms.functional as TF
import random
from termcolor import colored

class TransformsFixRotation(nn.Module):
    r"""
    Rotate by one of the given angles.

    Example: `rotation_transform = MyRotationTransform(angles=[-30, -15, 0, 15, 30])`
    """

    def __init__(self, angles):
        super(TransformsFixRotation, self).__init__()
        if not isinstance(angles, Sequence):
            angles = [angles, ]
        self.angles = angles

    def forward(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(angles={self.angles})"

VIL_COLORS = [[0, 0, 0],
              [0.30196078431372547, 0.30196078431372547, 0.30196078431372547],
              [0.1568627450980392, 0.7450980392156863, 0.1568627450980392],
              [0.09803921568627451, 0.5882352941176471, 0.09803921568627451],
              [0.0392156862745098, 0.4117647058823529, 0.0392156862745098],
              [0.0392156862745098, 0.29411764705882354, 0.0392156862745098],
              [0.9607843137254902, 0.9607843137254902, 0.0],
              [0.9294117647058824, 0.6745098039215687, 0.0],
              [0.9411764705882353, 0.43137254901960786, 0.0],
              [0.6274509803921569, 0.0, 0.0],
              [0.9058823529411765, 0.0, 1.0]]

VIL_LEVELS = [0.0, 16.0, 31.0, 59.0, 74.0, 100.0, 133.0, 160.0, 181.0, 219.0, 255.0]

def vil_cmap(encoded=True):
    cols = deepcopy(VIL_COLORS)
    lev = deepcopy(VIL_LEVELS)
    # Exactly the same error occurs in the original implementation (https://github.com/MIT-AI-Accelerator/neurips-2020-sevir/blob/master/src/display/display.py).
    # ValueError: There are 10 color bins including extensions, but ncolors = 9; ncolors must equal or exceed the number of bins
    # We can not replicate the visualization in notebook (https://github.com/MIT-AI-Accelerator/neurips-2020-sevir/blob/master/notebooks/AnalyzeNowcast.ipynb) without error.
    nil = cols.pop(0)
    under = cols[0]
    # over = cols.pop()
    over = cols[-1]
    cmap = ListedColormap(cols)
    cmap.set_bad(nil)
    cmap.set_under(under)
    cmap.set_over(over)
    norm = BoundaryNorm(lev, cmap.N)
    vmin, vmax = None, None
    return cmap, norm, vmin, vmax

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

default_exps_dir = os.path.abspath(os.path.join(root_dir, "experiments"))

default_dataset_dir = os.path.abspath(os.path.join(root_dir, "datasets"))
default_dataset_sevir_dir = os.path.abspath(os.path.join(default_dataset_dir, "sevir"))


# SEVIR Dataset constants
SEVIR_DATA_TYPES = ['vis', 'ir069', 'ir107', 'vil', 'lght']
SEVIR_RAW_DTYPES = {'vis': np.int16,
                    'ir069': np.int16,
                    'ir107': np.int16,
                    'vil': np.uint8,
                    'lght': np.int16}
LIGHTING_FRAME_TIMES = np.arange(- 120.0, 125.0, 5) * 60
SEVIR_DATA_SHAPE = {'lght': (48, 48), }
PREPROCESS_SCALE_SEVIR = {'vis': 1,  # Not utilized in original paper
                          'ir069': 1 / 1174.68,
                          'ir107': 1 / 2562.43,
                          'vil': 1 / 47.54,
                          'lght': 1 / 0.60517}
PREPROCESS_OFFSET_SEVIR = {'vis': 0,  # Not utilized in original paper
                           'ir069': 3683.58,
                           'ir107': 1552.80,
                           'vil': - 33.44,
                           'lght': - 0.02990}
PREPROCESS_SCALE_01 = {'vis': 1,
                       'ir069': 1,
                       'ir107': 1,
                       'vil': 1 / 255,  # currently the only one implemented
                       'lght': 1}
PREPROCESS_OFFSET_01 = {'vis': 0,
                        'ir069': 0,
                        'ir107': 0,
                        'vil': 0,  # currently the only one implemented
                        'lght': 0}

HMF_COLORS = np.array([
    [82, 82, 82],
    [252, 141, 89],
    [255, 255, 191],
    [145, 191, 219]
]) / 255

THRESHOLDS = (0, 16, 74, 133, 160, 181, 219, 255)
# sevir
SEVIR_CATALOG = os.path.join(default_dataset_sevir_dir, "CATALOG.csv")
SEVIR_DATA_DIR = os.path.join(default_dataset_sevir_dir, "data")
SEVIR_RAW_SEQ_LEN = 49

def path_splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def change_layout(data,
                  in_layout='NHWT', out_layout='NHWT',
                  ret_contiguous=False):
    in_layout = " ".join(in_layout.replace("C", "1"))
    out_layout = " ".join(out_layout.replace("C", "1"))
    data = rearrange(data, f"{in_layout} -> {out_layout}")
    if ret_contiguous:
        if isinstance(data, np.ndarray):
            data = np.ascontiguousarray(data)
        elif isinstance(data, torch.Tensor):
            data = data.contiguous()
        else:
            raise ValueError
    return data


class SEVIRDataLoader:
    r"""
    DataLoader that loads SEVIR sequences, and spilts each event
    into segments according to specified sequence length.

    Event Frames:
        [-----------------------raw_seq_len----------------------]
        [-----seq_len-----]
        <--stride-->[-----seq_len-----]
                    <--stride-->[-----seq_len-----]
                                        ...
    """
    def __init__(self,
                 data_types: Optional[Sequence[str]] = None,
                 seq_len: int = 49,
                 raw_seq_len: int = 49,
                 sample_mode: str = 'sequent',
                 stride: int = 12,
                 batch_size: int = 1,
                 layout: str = 'NHWT',
                 num_shard: int = 1,
                 rank: int = 0,
                 split_mode: str = "uneven",
                 sevir_catalog: Optional[Union[str, pd.DataFrame]] = None,
                 sevir_data_dir: Optional[str] = None,
                 start_date: Optional[datetime.datetime] = None,
                 end_date: Optional[datetime.datetime] = None,
                 datetime_filter=None,
                 catalog_filter='default',
                 shuffle: bool = False,
                 shuffle_seed: int = 1,
                 output_type=np.float32,
                 preprocess: bool = True,
                 rescale_method: str = '01',
                 downsample_dict: Optional[Dict[str, Sequence[int]]] = None,
                 verbose: bool = False):
        r"""
        Parameters
        ----------
        data_types
            A subset of SEVIR_DATA_TYPES.
        seq_len
            The length of the data sequences. Should be smaller than the max length raw_seq_len.
        raw_seq_len
            The length of the raw data sequences.
        sample_mode
            'random' or 'sequent'
        stride
            Useful when sample_mode == 'sequent'
            stride must not be smaller than out_len to prevent data leakage in testing.
        batch_size
            Number of sequences in one batch.
        layout
            str: consists of batch_size 'N', seq_len 'T', channel 'C', height 'H', width 'W'
            The layout of sampled data. Raw data layout is 'NHWT'.
            valid layout: 'NHWT', 'NTHW', 'NTCHW', 'TNHW', 'TNCHW'.
        num_shard
            Split the whole dataset into num_shard parts for distributed training.
        rank
            Rank of the current process within num_shard.
        split_mode: str
            if 'ceil', all `num_shard` dataloaders have the same length = ceil(total_len / num_shard).
            Different dataloaders may have some duplicated data batches, if the total size of datasets is not divided by num_shard.
            if 'floor', all `num_shard` dataloaders have the same length = floor(total_len / num_shard).
            The last several data batches may be wasted, if the total size of datasets is not divided by num_shard.
            if 'uneven', the last datasets has larger length when the total length is not divided by num_shard.
            The uneven split leads to synchronization error in dist.all_reduce() or dist.barrier().
            See related issue: https://github.com/pytorch/pytorch/issues/33148
            Notice: this also affects the behavior of `self.use_up`.
        sevir_catalog
            Name of SEVIR catalog CSV file.
        sevir_data_dir
            Directory path to SEVIR data.
        start_date
            Start time of SEVIR samples to generate.
        end_date
            End time of SEVIR samples to generate.
        datetime_filter
            function
            Mask function applied to time_utc column of catalog (return true to keep the row).
            Pass function of the form   lambda t : COND(t)
            Example:  lambda t: np.logical_and(t.dt.hour>=13,t.dt.hour<=21)  # Generate only day-time events
        catalog_filter
            function or None or 'default'
            Mask function applied to entire catalog dataframe (return true to keep row).
            Pass function of the form lambda catalog:  COND(catalog)
            Example:  lambda c:  [s[0]=='S' for s in c.id]   # Generate only the 'S' events
        shuffle
            bool, If True, data samples are shuffled before each epoch.
        shuffle_seed
            int, Seed to use for shuffling.
        output_type
            np.dtype, dtype of generated tensors
        preprocess
            bool, If True, self.preprocess_data_dict(data_dict) is called before each sample generated
        downsample_dict:
            dict, downsample_dict.keys() == data_types. downsample_dict[key] is a Sequence of (t_factor, h_factor, w_factor),
            representing the downsampling factors of all dimensions.
        verbose
            bool, verbose when opening raw data files
        """
        super(SEVIRDataLoader, self).__init__()
        if sevir_catalog is None:
            sevir_catalog = SEVIR_CATALOG
        if sevir_data_dir is None:
            sevir_data_dir = SEVIR_DATA_DIR
        if data_types is None:
            data_types = SEVIR_DATA_TYPES
        else:
            assert set(data_types).issubset(SEVIR_DATA_TYPES)

        # configs which should not be modified
        self._dtypes = SEVIR_RAW_DTYPES
        self.lght_frame_times = LIGHTING_FRAME_TIMES
        self.data_shape = SEVIR_DATA_SHAPE

        self.raw_seq_len = raw_seq_len
        assert seq_len <= self.raw_seq_len, f'seq_len must not be larger than raw_seq_len = {raw_seq_len}, got {seq_len}.'
        self.seq_len = seq_len
        assert sample_mode in ['random', 'sequent'], f'Invalid sample_mode = {sample_mode}, must be \'random\' or \'sequent\'.'
        self.sample_mode = sample_mode
        self.stride = stride
        self.batch_size = batch_size
        valid_layout = ('NHWT', 'NTHW', 'NTCHW', 'NTHWC', 'TNHW', 'TNCHW')
        if layout not in valid_layout:
            raise ValueError(f'Invalid layout = {layout}! Must be one of {valid_layout}.')
        self.layout = layout
        self.num_shard = num_shard
        self.rank = rank
        valid_split_mode = ('ceil', 'floor', 'uneven')
        if split_mode not in valid_split_mode:
            raise ValueError(f'Invalid split_mode: {split_mode}! Must be one of {valid_split_mode}.')
        self.split_mode = split_mode
        self._samples = None
        self._hdf_files = {}
        self.data_types = data_types
        if isinstance(sevir_catalog, str):
            self.catalog = pd.read_csv(sevir_catalog, parse_dates=['time_utc'], low_memory=False)
        else:
            self.catalog = sevir_catalog
        self.sevir_data_dir = sevir_data_dir
        self.datetime_filter = datetime_filter
        self.catalog_filter = catalog_filter
        self.start_date = start_date
        self.end_date = end_date
        self.shuffle = shuffle
        self.shuffle_seed = int(shuffle_seed)
        self.output_type = output_type
        self.preprocess = preprocess
        self.downsample_dict = downsample_dict
        self.rescale_method = rescale_method
        self.verbose = verbose

        if self.start_date is not None:
            self.catalog = self.catalog[self.catalog.time_utc > self.start_date]
        if self.end_date is not None:
            self.catalog = self.catalog[self.catalog.time_utc <= self.end_date]
        if self.datetime_filter:
            self.catalog = self.catalog[self.datetime_filter(self.catalog.time_utc)]

        if self.catalog_filter is not None:
            if self.catalog_filter == 'default':
                self.catalog_filter = lambda c: c.pct_missing == 0
            if callable(self.catalog_filter):
                self.catalog = self.catalog[self.catalog_filter(self.catalog)]
        self._compute_samples()
        self._open_files(verbose=self.verbose)
        self.reset()

    def _compute_samples(self):
        """
        Computes the list of samples in catalog to be used. This sets self._samples
        """
        if self.catalog is None or self.data_types is None:
            print(colored("Catalog or data_types is None", "red"))
            return
        # locate all events containing colocated data_types
        imgt = self.data_types
        imgts = set(imgt)
        filtcat = self.catalog[ np.logical_or.reduce([self.catalog.img_type==i for i in imgt]) ]
        # remove rows missing one or more requested img_types
        filtcat = filtcat.groupby('id').filter(lambda x: imgts.issubset(set(x['img_type'])))
        # If there are repeated IDs, remove them (this is a bug in SEVIR)
        # TODO: is it necessary to keep one of them instead of deleting them all
        filtcat = filtcat.groupby('id').filter(lambda x: x.shape[0]==len(imgt))
        self._samples = filtcat.groupby('id').apply(lambda df: self._df_to_series(df,imgt) )
        assert self._samples is not None, colored("Samples is None", "red")
        if self.shuffle:
            self.shuffle_samples()

    def shuffle_samples(self):
        self._samples = self._samples.sample(frac=1, random_state=self.shuffle_seed)

    def _df_to_series(self, df, imgt):
        d = {}
        df = df.set_index('img_type')
        for i in imgt:
            s = df.loc[i]
            idx = s.file_index if i != 'lght' else s.id
            d.update({f'{i}_filename': [s.file_name],
                      f'{i}_index': [idx]})

        return pd.DataFrame(d)

    def _open_files(self, verbose=True):
        """
        Opens HDF files
        """
        imgt = self.data_types
        hdf_filenames = []
        if self.data_types is not None and self._samples is not None:
            for t in self.data_types:
                hdf_filenames += list(np.unique( self._samples[f'{t}_filename'].values ))       
        self._hdf_files = {}
        for f in hdf_filenames:
            if verbose:
                print('Opening HDF5 file for reading', f)
            self._hdf_files[f] = h5py.File(self.sevir_data_dir + '/' + f, 'r')

    def close(self):
        """
        Closes all open file handles
        """
        for f in self._hdf_files:
            self._hdf_files[f].close()
        self._hdf_files = {}

    @property
    def num_seq_per_event(self):
        return 1 + (self.raw_seq_len - self.seq_len) // self.stride

    @property
    def total_num_seq(self):
        """
        The total number of sequences within each shard.
        Notice that it is not the product of `self.num_seq_per_event` and `self.total_num_event`.
        """
        return int(self.num_seq_per_event * self.num_event)

    @property
    def total_num_event(self):
        """
        The total number of events in the whole dataset, before split into different shards.
        """
        if self._samples is not None:
            return int(self._samples.shape[0])
        return 0

    @property
    def start_event_idx(self):
        """
        The event idx used in certain rank should satisfy event_idx >= start_event_idx
        """
        return self.total_num_event // self.num_shard * self.rank

    @property
    def end_event_idx(self):
        """
        The event idx used in certain rank should satisfy event_idx < end_event_idx

        """
        if self.split_mode == 'ceil':
            _last_start_event_idx = self.total_num_event // self.num_shard * (self.num_shard - 1)
            _num_event = self.total_num_event - _last_start_event_idx
            return self.start_event_idx + _num_event
        elif self.split_mode == 'floor':
            return self.total_num_event // self.num_shard * (self.rank + 1)
        else:  # self.split_mode == 'uneven':
            if self.rank == self.num_shard - 1:  # the last process
                return self.total_num_event
            else:
                return self.total_num_event // self.num_shard * (self.rank + 1)

    @property
    def num_event(self):
        """
        The number of events split into each rank
        """
        return self.end_event_idx - self.start_event_idx

    def _read_data(self, row, data):
        """
        Iteratively read data into data dict. Finally data[imgt] gets shape (batch_size, height, width, raw_seq_len).

        Parameters
        ----------
        row
            A series with fields IMGTYPE_filename, IMGTYPE_index, IMGTYPE_time_index.
        data
            Dict, data[imgt] is a data tensor with shape = (tmp_batch_size, height, width, raw_seq_len).

        Returns
        -------
        data
            Updated data. Updated shape = (tmp_batch_size + 1, height, width, raw_seq_len).
        """
        imgtyps = np.unique([x.split('_')[0] for x in list(row.keys())])
        for t in imgtyps:
            fname = row[f'{t}_filename']
            idx = row[f'{t}_index']
            t_slice = slice(0, None)
            # Need to bin lght counts into grid
            if t == 'lght':
                lght_data = self._hdf_files[fname][idx][:]
                data_i = self._lght_to_grid(lght_data, t_slice)
            else:
                data_i = self._hdf_files[fname][t][idx:idx + 1, :, :, t_slice]
            data[t] = np.concatenate((data[t], data_i), axis=0) if (t in data) else data_i

        return data

    def _lght_to_grid(self, data, t_slice=slice(0, None)):
        """
        Converts Nx5 lightning data matrix into a 2D grid of pixel counts
        """
        # out_size = (48,48,len(self.lght_frame_times)-1) if isinstance(t_slice,(slice,)) else (48,48)
        out_size = (*self.data_shape['lght'], len(self.lght_frame_times)) if t_slice.stop is None else (*self.data_shape['lght'], 1)
        if data.shape[0] == 0:
            return np.zeros((1,) + out_size, dtype=np.float32)

        # filter out points outside the grid
        x, y = data[:, 3], data[:, 4]
        m = np.logical_and.reduce([x >= 0, x < out_size[0], y >= 0, y < out_size[1]])
        data = data[m, :]
        if data.shape[0] == 0:
            return np.zeros((1,) + out_size, dtype=np.float32)

        # Filter/separate times
        t = data[:, 0]
        if t_slice.stop is not None:  # select only one time bin
            if t_slice.stop > 0:
                if t_slice.stop < len(self.lght_frame_times):
                    tm = np.logical_and(t >= self.lght_frame_times[t_slice.stop - 1],
                                        t < self.lght_frame_times[t_slice.stop])
                else:
                    tm = t >= self.lght_frame_times[-1]
            else:  # special case:  frame 0 uses lght from frame 1
                tm = np.logical_and(t >= self.lght_frame_times[0], t < self.lght_frame_times[1])
            # tm=np.logical_and( (t>=FRAME_TIMES[t_slice],t<FRAME_TIMES[t_slice+1]) )

            data = data[tm, :]
            z = np.zeros(data.shape[0], dtype=np.int64)
        else:  # compute z coordinate based on bin location times
            z = np.digitize(t, self.lght_frame_times) - 1
            z[z == -1] = 0  # special case:  frame 0 uses lght from frame 1

        x = data[:, 3].astype(np.int64)
        y = data[:, 4].astype(np.int64)

        k = np.ravel_multi_index(np.array([y, x, z]), out_size)
        n = np.bincount(k, minlength=np.prod(out_size))
        return np.reshape(n, out_size).astype(np.int16)[np.newaxis, :]

    def save_downsampled_dataset(self,
                                 save_dir: str,
                                 downsample_dict: Dict[str, Sequence[int]],
                                 verbose=True):
        """
        Parameters
        ----------
        save_dir
        downsample_dict:    Dict[str, Sequence[int]]
            Notice that this is different from `self.downsample_dict`, which is used during runtime.
        """
        import os
        from skimage.measure import block_reduce
        assert not os.path.exists(save_dir), f"save_dir {save_dir} already exists!"
        os.makedirs(save_dir)
        for fname, hdf_file in self._hdf_files.items():
            if verbose:
                print(f"Downsampling data in {fname}.")
            data_type = path_splitall(fname)[0]
            if data_type == 'lght':
                # TODO: how to get idx?
                raise NotImplementedError
                # lght_data = self._hdf_files[fname][idx][:]
                # t_slice = slice(0, None)
                # data_i = self._lght_to_grid(lght_data, t_slice)
            else:
                data_i = self._hdf_files[fname][data_type]
            # Downsample t
            t_slice = [slice(None, None), ] * 4
            t_slice[-1] = slice(None, None, downsample_dict[data_type][0])  # layout = 'NHWT'
            data_i = data_i[tuple(t_slice)]
            # Downsample h, w
            data_i = block_reduce(data_i,
                                  block_size=(1, *downsample_dict[data_type][1:], 1),
                                  func=np.max)
            # Save as new .h5 file
            new_file_path = os.path.join(save_dir, fname)
            if not os.path.exists(os.path.dirname(new_file_path)):
                os.makedirs(os.path.dirname(new_file_path))
            # Create dataset
            with h5py.File(new_file_path, 'w') as hf:
                hf.create_dataset(
                    data_type, data=data_i,
                    maxshape=(None, *data_i.shape[1:]))

    @property
    def sample_count(self):
        """
        Record how many times self.__next__() is called.
        """
        return self._sample_count

    def inc_sample_count(self):
        self._sample_count += 1

    @property
    def curr_event_idx(self):
        return self._curr_event_idx

    @property
    def curr_seq_idx(self):
        """
        Used only when self.sample_mode == 'sequent'
        """
        return self._curr_seq_idx

    def set_curr_event_idx(self, val):
        self._curr_event_idx = val

    def set_curr_seq_idx(self, val):
        """
        Used only when self.sample_mode == 'sequent'
        """
        self._curr_seq_idx = val

    def reset(self, shuffle: Optional[bool] = None):
        self.set_curr_event_idx(val=self.start_event_idx)
        self.set_curr_seq_idx(0)
        self._sample_count = 0
        if shuffle is None:
            shuffle = self.shuffle
        if shuffle:
            self.shuffle_samples()

    def __len__(self):
        """
        Used only when self.sample_mode == 'sequent'
        """
        return self.total_num_seq // self.batch_size

    @property
    def use_up(self):
        """
        Check if dataset is used up in 'sequent' mode.
        """
        if self.sample_mode == 'random':
            return False
        else:   # self.sample_mode == 'sequent'
            # compute the remaining number of sequences in current event
            curr_event_remain_seq = self.num_seq_per_event - self.curr_seq_idx
            all_remain_seq = curr_event_remain_seq + (
                        self.end_event_idx - self.curr_event_idx - 1) * self.num_seq_per_event
            if self.split_mode == "floor":
                # This approach does not cover all available data, but avoid dealing with masks
                return all_remain_seq < self.batch_size
            else:
                return all_remain_seq <= 0

    def _load_event_batch(self, event_idx, event_batch_size):
        """
        Loads a selected batch of events (not batch of sequences) into memory.

        Parameters
        ----------
        idx
        event_batch_size
            event_batch[i] = all_type_i_available_events[idx:idx + event_batch_size]
        Returns
        -------
        event_batch
            list of event batches.
            event_batch[i] is the event batch of the i-th data type.
            Each event_batch[i] is a np.ndarray with shape = (event_batch_size, height, width, raw_seq_len)
        """
        event_idx_slice_end = event_idx + event_batch_size
        pad_size = 0
        if event_idx_slice_end > self.end_event_idx:
            pad_size = event_idx_slice_end - self.end_event_idx
            event_idx_slice_end = self.end_event_idx
        pd_batch = self._samples.iloc[event_idx:event_idx_slice_end]
        data = {}
        for index, row in pd_batch.iterrows():
            data = self._read_data(row, data)
        if pad_size > 0:
            event_batch = []
            for t in self.data_types:
                pad_shape = [pad_size, ] + list(data[t].shape[1:])
                data_pad = np.concatenate((data[t].astype(self.output_type),
                                           np.zeros(pad_shape, dtype=self.output_type)),
                                          axis=0)
                event_batch.append(data_pad)
        else:
            event_batch = [data[t].astype(self.output_type) for t in self.data_types]
        return event_batch

    def __iter__(self):
        return self

    def __next__(self):
        if self.sample_mode == 'random':
            self.inc_sample_count()
            ret_dict = self._random_sample()
        else:
            if self.use_up:
                raise StopIteration
            else:
                self.inc_sample_count()
                ret_dict = self._sequent_sample()
        ret_dict = self.data_dict_to_tensor(data_dict=ret_dict,
                                            data_types=self.data_types)
        if self.preprocess:
            ret_dict = self.preprocess_data_dict(data_dict=ret_dict,
                                                 data_types=self.data_types,
                                                 layout=self.layout,
                                                 rescale=self.rescale_method)
        if self.downsample_dict is not None:
            ret_dict = self.downsample_data_dict(data_dict=ret_dict,
                                                 data_types=self.data_types,
                                                 factors_dict=self.downsample_dict,
                                                 layout=self.layout)
        return ret_dict

    def __getitem__(self, index):
        data_dict = self._idx_sample(index=index)
        return data_dict

    @staticmethod
    def preprocess_data_dict(data_dict, data_types=None, layout='NHWT', rescale='01'):
        """
        Parameters
        ----------
        data_dict:  Dict[str, Union[np.ndarray, torch.Tensor]]
        data_types: Sequence[str]
            The data types that we want to rescale. This mainly excludes "mask" from preprocessing.
        layout: str
            consists of batch_size 'N', seq_len 'T', channel 'C', height 'H', width 'W'
        rescale:    str
            'sevir': use the offsets and scale factors in original implementation.
            '01': scale all values to range 0 to 1, currently only supports 'vil'
        Returns
        -------
        data_dict:  Dict[str, Union[np.ndarray, torch.Tensor]]
            preprocessed data
        """
        if rescale == 'sevir':
            scale_dict = PREPROCESS_SCALE_SEVIR
            offset_dict = PREPROCESS_OFFSET_SEVIR
        elif rescale == '01':
            scale_dict = PREPROCESS_SCALE_01
            offset_dict = PREPROCESS_OFFSET_01
        else:
            raise ValueError(f'Invalid rescale option: {rescale}.')
        if data_types is None:
            data_types = data_dict.keys()
        for key, data in data_dict.items():
            if key in data_types:
                if isinstance(data, np.ndarray):
                    data = data.astype(np.float32)
                elif isinstance(data, torch.Tensor):
                    data = data.float()
                else:
                    raise TypeError
                data = change_layout(data=scale_dict[key] * (data + offset_dict[key]),
                                     in_layout='NHWT',
                                     out_layout=layout)
                data_dict[key] = data
        return data_dict

    @staticmethod
    def process_data_dict_back(data_dict, data_types=None, rescale='01'):
        """
        Parameters
        ----------
        data_dict
            each data_dict[key] is a torch.Tensor.
        rescale
            str:
                'sevir': data are scaled using the offsets and scale factors in original implementation.
                '01': data are all scaled to range 0 to 1, currently only supports 'vil'
        Returns
        -------
        data_dict
            each data_dict[key] is the data processed back in torch.Tensor.
        """
        if rescale == 'sevir':
            scale_dict = PREPROCESS_SCALE_SEVIR
            offset_dict = PREPROCESS_OFFSET_SEVIR
        elif rescale == '01':
            scale_dict = PREPROCESS_SCALE_01
            offset_dict = PREPROCESS_OFFSET_01
        else:
            raise ValueError(f'Invalid rescale option: {rescale}.')
        if data_types is None:
            data_types = list(data_dict.keys())
        for key in data_types:
            if key in data_dict:
                data = data_dict[key]
                data = data.float() / scale_dict[key] - offset_dict[key]
                data_dict[key] = data
        return data_dict

    @staticmethod
    def data_dict_to_tensor(data_dict, data_types=None):
        """
        Convert each element in data_dict to torch.Tensor (copy without grad).
        """
        ret_dict = {}
        if data_types is None:
            data_types = list(data_dict.keys())
        for key, data in data_dict.items():
            if key in data_types:
                if isinstance(data, torch.Tensor):
                    ret_dict[key] = data.detach().clone()
                elif isinstance(data, np.ndarray):
                    ret_dict[key] = torch.from_numpy(data)
                else:
                    raise ValueError(f"Invalid data type: {type(data)}. Should be torch.Tensor or np.ndarray")
            else:   # key == "mask"
                ret_dict[key] = data
        return ret_dict

    @staticmethod
    def downsample_data_dict(data_dict, data_types=None, factors_dict=None, layout='NHWT'):
        """
        Parameters
        ----------
        data_dict:  Dict[str, Union[np.array, torch.Tensor]]
        factors_dict:   Optional[Dict[str, Sequence[int]]]
            each element `factors` is a Sequence of int, representing (t_factor, h_factor, w_factor)

        Returns
        -------
        downsampled_data_dict:  Dict[str, torch.Tensor]
            Modify on a deep copy of data_dict instead of directly modifying the original data_dict
        """
        if factors_dict is None:
            factors_dict = {}
        if data_types is None:
            data_types = data_dict.keys()
        downsampled_data_dict = SEVIRDataLoader.data_dict_to_tensor(
            data_dict=data_dict,
            data_types=data_types)    # make a copy
        for key, data in data_dict.items():
            factors = factors_dict.get(key, None)
            if factors is not None:
                downsampled_data_dict[key] = change_layout(
                    data=downsampled_data_dict[key],
                    in_layout=layout,
                    out_layout='NTHW')
                # downsample t dimension
                t_slice = [slice(None, None), ] * 4
                t_slice[1] = slice(None, None, factors[0])
                downsampled_data_dict[key] = downsampled_data_dict[key][tuple(t_slice)]
                # downsample spatial dimensions
                downsampled_data_dict[key] = avg_pool2d(
                    input=downsampled_data_dict[key],
                    kernel_size=(factors[1], factors[2]))

                downsampled_data_dict[key] = change_layout(
                    data=downsampled_data_dict[key],
                    in_layout='NTHW',
                    out_layout=layout)

        return downsampled_data_dict

    def _random_sample(self):
        """
        Returns
        -------
        ret_dict
            dict. ret_dict.keys() == self.data_types.
            If self.preprocess == False:
                ret_dict[imgt].shape == (batch_size, height, width, seq_len)
        """
        num_sampled = 0
        event_idx_list = nprand.randint(low=self.start_event_idx,
                                        high=self.end_event_idx,
                                        size=self.batch_size)
        seq_idx_list = nprand.randint(low=0,
                                      high=self.num_seq_per_event,
                                      size=self.batch_size)
        seq_slice_list = [slice(seq_idx * self.stride,
                                seq_idx * self.stride + self.seq_len)
                          for seq_idx in seq_idx_list]
        ret_dict = {}
        while num_sampled < self.batch_size:
            event = self._load_event_batch(event_idx=event_idx_list[num_sampled],
                                           event_batch_size=1)
            for imgt_idx, imgt in enumerate(self.data_types):
                sampled_seq = event[imgt_idx][[0, ], :, :, seq_slice_list[num_sampled]]  # keep the dim of batch_size for concatenation
                if imgt in ret_dict:
                    ret_dict[imgt] = np.concatenate((ret_dict[imgt], sampled_seq),
                                                    axis=0)
                else:
                    ret_dict.update({imgt: sampled_seq})
        return ret_dict

    def _sequent_sample(self):
        """
        Returns
        -------
        ret_dict:   Dict
            `ret_dict.keys()` contains `self.data_types`.
            `ret_dict["mask"]` is a list of bool, indicating if the data entry is real or padded.
            If self.preprocess == False:
                ret_dict[imgt].shape == (batch_size, height, width, seq_len)
        """
        assert not self.use_up, 'Data loader used up! Reset it to reuse.'
        event_idx = self.curr_event_idx
        seq_idx = self.curr_seq_idx
        num_sampled = 0
        sampled_idx_list = []   # list of (event_idx, seq_idx) records
        while num_sampled < self.batch_size:
            sampled_idx_list.append({'event_idx': event_idx,
                                     'seq_idx': seq_idx})
            seq_idx += 1
            if seq_idx >= self.num_seq_per_event:
                event_idx += 1
                seq_idx = 0
            num_sampled += 1

        start_event_idx = sampled_idx_list[0]['event_idx']
        event_batch_size = sampled_idx_list[-1]['event_idx'] - start_event_idx + 1

        event_batch = self._load_event_batch(event_idx=start_event_idx,
                                             event_batch_size=event_batch_size)
        ret_dict = {"mask": []}
        all_no_pad_flag = True
        for sampled_idx in sampled_idx_list:
            batch_slice = [sampled_idx['event_idx'] - start_event_idx, ]  # use [] to keepdim
            seq_slice = slice(sampled_idx['seq_idx'] * self.stride,
                              sampled_idx['seq_idx'] * self.stride + self.seq_len)
            for imgt_idx, imgt in enumerate(self.data_types):
                sampled_seq = event_batch[imgt_idx][batch_slice, :, :, seq_slice]
                if imgt in ret_dict:
                    ret_dict[imgt] = np.concatenate((ret_dict[imgt], sampled_seq),
                                                    axis=0)
                else:
                    ret_dict.update({imgt: sampled_seq})
            # add mask
            no_pad_flag = sampled_idx['event_idx'] < self.end_event_idx
            if not no_pad_flag:
                all_no_pad_flag = False
            ret_dict["mask"].append(no_pad_flag)
        if all_no_pad_flag:
            # if there is no padded data items at all, set `ret_dict["mask"] = None` for convenience.
            del ret_dict["mask"]
        # update current idx
        self.set_curr_event_idx(event_idx)
        self.set_curr_seq_idx(seq_idx)
        return ret_dict

    def _idx_sample(self, index):
        """
        Parameters
        ----------
        index
            The index of the batch to sample.
        Returns
        -------
        ret_dict
            dict. ret_dict.keys() == self.data_types.
            If self.preprocess == False:
                ret_dict[imgt].shape == (batch_size, height, width, seq_len)
        """
        event_idx = (index * self.batch_size) // self.num_seq_per_event
        seq_idx = (index * self.batch_size) % self.num_seq_per_event
        num_sampled = 0
        sampled_idx_list = []  # list of (event_idx, seq_idx) records
        while num_sampled < self.batch_size:
            sampled_idx_list.append({'event_idx': event_idx,
                                     'seq_idx': seq_idx})
            seq_idx += 1
            if seq_idx >= self.num_seq_per_event:
                event_idx += 1
                seq_idx = 0
            num_sampled += 1

        start_event_idx = sampled_idx_list[0]['event_idx']
        event_batch_size = sampled_idx_list[-1]['event_idx'] - start_event_idx + 1

        event_batch = self._load_event_batch(event_idx=start_event_idx,
                                             event_batch_size=event_batch_size)
        ret_dict = {}
        for sampled_idx in sampled_idx_list:
            batch_slice = [sampled_idx['event_idx'] - start_event_idx, ]  # use [] to keepdim
            seq_slice = slice(sampled_idx['seq_idx'] * self.stride,
                              sampled_idx['seq_idx'] * self.stride + self.seq_len)
            for imgt_idx, imgt in enumerate(self.data_types):
                sampled_seq = event_batch[imgt_idx][batch_slice, :, :, seq_slice]
                if imgt in ret_dict:
                    ret_dict[imgt] = np.concatenate((ret_dict[imgt], sampled_seq),
                                                    axis=0)
                else:
                    ret_dict.update({imgt: sampled_seq})

        ret_dict = self.data_dict_to_tensor(data_dict=ret_dict,
                                            data_types=self.data_types)
        if self.preprocess:
            ret_dict = self.preprocess_data_dict(data_dict=ret_dict,
                                                 data_types=self.data_types,
                                                 layout=self.layout,
                                                 rescale=self.rescale_method)

        if self.downsample_dict is not None:
            ret_dict = self.downsample_data_dict(data_dict=ret_dict,
                                                 data_types=self.data_types,
                                                 factors_dict=self.downsample_dict,
                                                 layout=self.layout)
        return ret_dict

#################################################################################################################################

class SEVIRTorchDataset(TorchDataset):

    orig_dataloader_layout = "NHWT"
    orig_dataloader_squeeze_layout = orig_dataloader_layout.replace("N", "")
    aug_layout = "THW"

    def __init__(self,
                 seq_len: int = 25,
                 raw_seq_len: int = 49,
                 sample_mode: str = "sequent",
                 stride: int = 12,
                 layout: str = "THWC",
                 split_mode: str = "uneven",
                 sevir_catalog: Optional[Union[str, pd.DataFrame]] = None,
                 sevir_data_dir: Optional[str] = None,
                 start_date: Optional[datetime.datetime] = None,
                 end_date: Optional[datetime.datetime] = None,
                 datetime_filter = None,
                 catalog_filter = "default",
                 shuffle: bool = False,
                 shuffle_seed: int = 1,
                 output_type = np.float32,
                 preprocess: bool = True,
                 rescale_method: str = "01",
                 verbose: bool = False,
                 aug_mode: str = "0",
                 ret_contiguous: bool = True):
        super(SEVIRTorchDataset, self).__init__()
        self.layout = layout.replace("C", "1")
        self.ret_contiguous = ret_contiguous
        self.sevir_dataloader = SEVIRDataLoader(
            data_types=["vil", ],
            seq_len=seq_len,
            raw_seq_len=raw_seq_len,
            sample_mode=sample_mode,
            stride=stride,
            batch_size=1,
            layout=self.orig_dataloader_layout,
            num_shard=1,
            rank=0,
            split_mode=split_mode,
            sevir_catalog=sevir_catalog,
            sevir_data_dir=sevir_data_dir,
            start_date=start_date,
            end_date=end_date,
            datetime_filter=datetime_filter,
            catalog_filter=catalog_filter,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            output_type=output_type,
            preprocess=preprocess,
            rescale_method=rescale_method,
            downsample_dict=None,
            verbose=verbose)
        self.aug_mode = aug_mode
        if aug_mode == "0":
            self.aug = lambda x:x
        elif aug_mode == "1":
            self.aug = nn.Sequential(
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=180),
            )
        elif aug_mode == "2":
            self.aug = nn.Sequential(
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                TransformsFixRotation(angles=[0, 90, 180, 270]),
            )
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        data_dict = self.sevir_dataloader._idx_sample(index=index)
        data = data_dict["vil"].squeeze(0)
        if self.aug_mode != "0":
            data = rearrange(data, f"{' '.join(self.orig_dataloader_squeeze_layout)} -> {' '.join(self.aug_layout)}")
            data = self.aug(data)
            data = rearrange(data, f"{' '.join(self.aug_layout)} -> {' '.join(self.layout)}")
        else:
            data = rearrange(data, f"{' '.join(self.orig_dataloader_squeeze_layout)} -> {' '.join(self.layout)}")
        if self.ret_contiguous:
            return data.contiguous()
        else:
            return data

    def __len__(self):
        return self.sevir_dataloader.__len__()


class SEVIRLightningDataModule(LightningDataModule):

    def __init__(self, cfg: DictConfig):
        super(SEVIRLightningDataModule, self).__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.seq_len = cfg.seq_len
        self.sample_mode = cfg.sample_mode
        self.stride = cfg.stride
        assert cfg.layout[0] == "N"
        self.layout = cfg.layout.replace("N", "")
        self.output_type = np.float32
        self.preprocess = cfg.preprocess
        self.rescale_method = cfg.rescale_method
        self.verbose = cfg.get("verbose", False)
        self.aug_mode = cfg.aug_mode
        self.ret_contiguous = cfg.ret_contiguous
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        self.seed = cfg.seed

        self.sevir_dir = cfg.data_dir
    
        self.catalog_path = os.path.join(self.sevir_dir, "CATALOG.csv")
        self.raw_data_dir = os.path.join(self.sevir_dir, "data")
        self.raw_seq_len = 49
        self.interval_real_time = 5
        self.img_height = 384
        self.img_width = 384

        # train val test split
        def _parse_date(date_str):
            if date_str is None:
                return None
            return datetime.datetime.strptime(date_str, '%Y-%m-%d')
        self.start_date = _parse_date(cfg.start_date) \
            if "start_date" in cfg else None
        self.train_test_split_date = _parse_date(cfg.train_test_split_date) \
            if "train_test_split_date" in cfg else None
        self.end_date = _parse_date(cfg.end_date) \
            if "end_date" in cfg else None
        self.val_ratio = cfg.val_ratio

    def setup(self, stage: Optional[str] = None) -> None:
        seed_everything(seed=self.seed)
        if stage in (None, "fit"):
            sevir_train_val = SEVIRTorchDataset(
                sevir_catalog=self.catalog_path,
                sevir_data_dir=self.raw_data_dir,
                raw_seq_len=self.raw_seq_len,
                split_mode="uneven",
                shuffle=True,
                seq_len=self.seq_len,
                stride=self.stride,
                sample_mode=self.sample_mode,
                layout=self.layout,
                start_date=self.start_date,
                end_date=self.train_test_split_date,
                output_type=self.output_type,
                preprocess=self.preprocess,
                rescale_method=self.rescale_method,
                verbose=self.verbose,
                aug_mode=self.aug_mode,
                ret_contiguous=self.ret_contiguous,)
            self.sevir_train, self.sevir_val = random_split(
                dataset=sevir_train_val,
                lengths=[1 - self.val_ratio, self.val_ratio],
                generator=torch.Generator().manual_seed(self.seed))
        if stage in (None, "test"):
            self.sevir_test = SEVIRTorchDataset(
                sevir_catalog=self.catalog_path,
                sevir_data_dir=self.raw_data_dir,
                raw_seq_len=self.raw_seq_len,
                split_mode="uneven",
                shuffle=False,
                seq_len=self.seq_len,
                stride=self.stride,
                sample_mode=self.sample_mode,
                layout=self.layout,
                start_date=self.train_test_split_date,
                end_date=self.end_date,
                output_type=self.output_type,
                preprocess=self.preprocess,
                rescale_method=self.rescale_method,
                verbose=self.verbose,
                aug_mode="0",
                ret_contiguous=self.ret_contiguous,)

    def train_dataloader(self):
        return DataLoader(self.sevir_train,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers, prefetch_factor=2, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.sevir_val,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers, prefetch_factor=2, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.sevir_test,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers, prefetch_factor=2, pin_memory=True)

    @property
    def num_train_samples(self):
        return len(self.sevir_train)

    @property
    def num_val_samples(self):
        return len(self.sevir_val)

    @property
    def num_test_samples(self):
        return len(self.sevir_test)

    @staticmethod
    def plot_hit_miss_fa(ax, y_true, y_pred, thres):
        mask = np.zeros_like(y_true)
        mask[np.logical_and(y_true >= thres, y_pred >= thres)] = 4
        mask[np.logical_and(y_true >= thres, y_pred < thres)] = 3
        mask[np.logical_and(y_true < thres, y_pred >= thres)] = 2
        mask[np.logical_and(y_true < thres, y_pred < thres)] = 1
        cmap = ListedColormap(HMF_COLORS)
        ax.imshow(mask, cmap=cmap)

    @staticmethod
    def plot_hit_miss_fa_all_thresholds(ax, y_true, y_pred, **unused_kwargs):
        fig = np.zeros(y_true.shape)
        y_true_idx = np.searchsorted(THRESHOLDS, y_true)
        y_pred_idx = np.searchsorted(THRESHOLDS, y_pred)
        fig[y_true_idx == y_pred_idx] = 4
        fig[y_true_idx > y_pred_idx] = 3
        fig[y_true_idx < y_pred_idx] = 2
        # do not count results in these not challenging areas.
        fig[np.logical_and(y_true < THRESHOLDS[1], y_pred < THRESHOLDS[1])] = 1
        cmap = ListedColormap(HMF_COLORS)
        ax.imshow(fig, cmap=cmap)

    @staticmethod
    def vis_sevir_seq(
            save_path,
            seq: Union[np.ndarray, Sequence[np.ndarray]],
            label: Union[str, Sequence[str]] = "pred",
            norm: Optional[Dict[str, float]] = None,
            interval_real_time: float = 10.0,  plot_stride=2,
            label_rotation=0,
            label_offset=(-0.06, 0.4),
            label_avg_int=False,
            fs=10,
            max_cols=10, ):
        """
        Parameters
        ----------
        seq:    Union[np.ndarray, Sequence[np.ndarray]]
            shape = (T, H, W). Float value 0-1 after `norm`.
        label:  Union[str, Sequence[str]]
            label for each sequence.
        norm:   Union[str, Dict[str, float]]
            seq_show = seq * norm['scale'] + norm['shift']
        interval_real_time: float
            The minutes of each plot interval
        max_cols: int
            The maximum number of columns in the figure.
        """

        def cmap_dict(s):
            assert s == "vil"
            return {'cmap': vil_cmap(encoded=True)[0],
                    'norm': vil_cmap(encoded=True)[1],
                    'vmin': vil_cmap(encoded=True)[2],
                    'vmax': vil_cmap(encoded=True)[3]}

        fontproperties = FontProperties()
        fontproperties.set_family('serif')
        # font.set_name('Times New Roman')
        fontproperties.set_size(fs)
        # font.set_weight("bold")

        if isinstance(seq, Sequence):
            seq_list = [ele.astype(np.float32) for ele in seq]
            assert isinstance(label, Sequence) and len(label) == len(seq)
            label_list = label
        elif isinstance(seq, np.ndarray):
            seq_list = [seq.astype(np.float32), ]
            assert isinstance(label, str)
            label_list = [label, ]
        else:
            raise NotImplementedError
        if label_avg_int:
            label_list = [f"{ele1}\nAvgInt = {np.mean(ele2): .3f}"
                        for ele1, ele2 in zip(label_list, seq_list)]
        # plot_stride
        seq_list = [ele[::plot_stride, ...] for ele in seq_list]
        seq_len_list = [len(ele) for ele in seq_list]

        max_len = max(seq_len_list) if seq_len_list else 0

        max_len = min(max_len, max_cols)
        seq_list_wrap = []
        label_list_wrap = []
        seq_len_list_wrap = []
        for i, (seq, label, seq_len) in enumerate(zip(seq_list, label_list, seq_len_list)):
            if max_len > 0:
                num_row = math.ceil(seq_len / max_len)
                for j in range(num_row):
                    slice_end = min(seq_len, (j + 1) * max_len)
                    seq_list_wrap.append(seq[j * max_len: slice_end])
                    if j == 0:
                        label_list_wrap.append(label)
                    else:
                        label_list_wrap.append("")
                    seq_len_list_wrap.append(min(seq_len - j * max_len, max_len))

        if norm is None:
            norm = {'scale': 255,
                    'shift': 0}
        nrows = len(seq_list_wrap)
        if nrows == 0 or max_len == 0:
            fig, ax = plt.subplots()
            plt.close(fig)
            return

        fig, ax = plt.subplots(nrows=nrows,
                            ncols=max_len,
                            figsize=(3 * max_len, 3 * nrows))
        if nrows == 1:
            ax = [ax]

        for i, (seq, label, seq_len) in enumerate(zip(seq_list_wrap, label_list_wrap, seq_len_list_wrap)):
            ax_row = ax[i]
            if not isinstance(ax_row, np.ndarray):
                ax_row = [ax_row]
            ax_row[0].set_ylabel(ylabel=label, fontproperties=fontproperties, rotation=label_rotation)
            ax_row[0].yaxis.set_label_coords(label_offset[0], label_offset[1])
            for j in range(0, max_len):
                if j < seq_len:
                    x = seq[j] * norm['scale'] + norm['shift']
                    ax_row[j].imshow(x, **cmap_dict('vil'))
                    if i == len(seq_list) - 1 and i > 0:  # the last row which is not the `in_seq`.
                        ax_row[j].set_title(f"Min {int(interval_real_time * (j + 1) * plot_stride)}",
                                            y=-0.25, fontproperties=fontproperties)
                else:
                    ax_row[j].axis('off')

        for i in range(nrows):
            ax_row = ax[i]
            if not isinstance(ax_row, np.ndarray):
                ax_row = [ax_row]
            for j in range(len(ax_row)):
                ax_row[j].xaxis.set_ticks([])
                ax_row[j].yaxis.set_ticks([])

        # Legend of thresholds
        num_thresh_legend = len(VIL_LEVELS) - 1
        legend_elements = [Patch(facecolor=VIL_COLORS[i],
                                label=f'{int(VIL_LEVELS[i - 1])}-{int(VIL_LEVELS[i])}')
                        for i in range(1, num_thresh_legend + 1)]
        ax[0][0].legend(handles=legend_elements, loc='center left',
                        bbox_to_anchor=(-1.2, -0.),
                        borderaxespad=0, frameon=False, fontsize='10')
        plt.subplots_adjust(hspace=0.05, wspace=0.05)
        plt.savefig(save_path)
        plt.close(fig)