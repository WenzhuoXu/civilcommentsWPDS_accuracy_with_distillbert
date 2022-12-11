import numpy as np
import os
import pickle

import torch
from torchvision.datasets.utils import download_url
import wilds
from wilds.datasets.fmow_dataset import FMoWDataset
from datasets.pds_dataset import WPDSDataset, WILDSSubDataset


class FMoWWPDS(WPDSDataset, FMoWDataset):
    _idx_lut = np.array([
        [120000,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,10000,9000,6900,6900,4100,4100,1500,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,1000,3100,3100,5900,5900,8500,10000,9000,6900,6900,4100,4100,1500,1500,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,1000,3100,3100,5900,5900,8500,8500,10000,9000,6900,6900,4100,4100,1500,1500,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1000,3100,3100,5900,5900,8500,8500,10000]
    ])

    _test_idx_lut = np.array([
        [1000,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1000,900,690,690,410,410,150,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,100,310,310,590,590,850,1000,900,690,690,410,410,150,150,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,100,310,310,590,590,850,850,1000,900,690,690,410,410,150,150,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,100,310,310,590,590,850,850,1000]
    ])

    def __init__(self, *args, magic=42, **kwargs):
        kwargs['version'] = '1.1'
        FMoWDataset.__init__(self, *args, **kwargs)
        self.year_id = 1
        assert self._metadata_fields[self.year_id] == 'year'

        self._n_batches = 25
        self._n_classes = 62
        self._n_groups = 5
        self._group_array = np.zeros((len(self),), dtype=int)
        for i in range(1, 5):
            self._group_array[self._metadata_array[:, 1] == i + 11] = i
        self._n_group_samples = [(self._group_array == t).sum() for t in range(self.num_groups)]

        self.idx_lut = self._idx_lut
        self.test_idx_lut = self._test_idx_lut

        self.idx_array = []
        np.random.seed(magic)
        for i in range(self.num_groups):
            idx = np.arange(len(self))
            idx = idx[self._group_array == i]
            idx = np.random.permutation(idx)
            self.idx_array.append(idx)

        # Regression set
        # Asia and Americas
        region_id = 0
        assert self._metadata_fields[region_id] == 'region'
        b = self._metadata_array[:, region_id].numpy()
        self.region_array = b
        a = self.idx_array[0][:120000]
        self.train_regression_idx = a[(b[a] == 0) | (b[a] == 3)]
        self.batch_zero_idx = a[(b[a] != 0) & (b[a] != 3)]

    def get_batch(self, t=None, transform=None, idx=None):
        if t is None or t != 0:
            return super().get_batch(t, transform, idx)
        return WILDSSubDataset(self, self.batch_zero_idx, transform)

    def get_train_regression(self, transform=None):
        indices = self.train_regression_idx
        dataset = WILDSSubDataset(self, indices, transform)
        return dataset     

    def get_test_regression(self, transform=None):
        a = self.idx_array[0][-10000:]
        b = self.region_array
        indices = a[(b[a] == 0) | (b[a] == 3)]
        dataset = WILDSSubDataset(self, indices, transform)
        return dataset        