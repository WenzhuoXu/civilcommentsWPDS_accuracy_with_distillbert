import numpy as np
import os
import pickle

import wilds
from wilds.datasets.poverty_dataset import PovertyMapDataset 
from datasets.pds_dataset import WPDSDataset, WILDSSubDataset


class PovertyWPDS(WPDSDataset, PovertyMapDataset):
    _idx_lut = np.array([
        [7029,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,800,720,720,552,552,328,328,120,120,0,0,0,0],
        [0,0,80,80,248,248,400,400,400,400,400,200,0,0],
        [0,0,0,0,0,0,72,72,280,280,400,600,800,800]
    ])

    _test_idx_lut = np.array([
        [100,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,100,90,90,69,69,41,41,15,15,0,0,0,0],
        [0,0,10,10,31,31,50,50,50,50,50,25,0,0],
        [0,0,0,0,0,0,9,9,35,35,50,75,100,100]
    ])

    def __init__(self, *args, magic=42, **kwargs):
        kwargs['version'] = '1.1'
        kwargs['no_nl'] = False
        kwargs['fold'] = 'A'
        kwargs['use_ood_val'] = True
        PovertyMapDataset.__init__(self, *args, **kwargs)

        self._n_batches = 14
        self._n_classes = 1 # Regression task
        self._n_groups = 4
        self._group_array = np.zeros((len(self),), dtype=int)
        b = self.metadata['year']
        b = b.to_numpy()
        self._group_array[b >= 2012] = 1
        self._group_array[b >= 2014] = 2
        self._group_array[b >= 2015] = 3
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
        # Urban areas
        c = self.metadata['urban'].to_numpy()
        a = self.idx_array[0]
        a = a[:-100]
        c = c[a]
        self.batch_zero_idx = a[(c == False)]
        a = a[(c == True)]
        self.train_regression_idx = a[:1500]
        self.test_regression_idx = a[1500:]

    def get_batch(self, t=None, transform=None, idx=None):
        if t is None or t != 0:
            return super().get_batch(t, transform, idx)
        return WILDSSubDataset(self, self.batch_zero_idx, transform)

    def get_train_regression(self, transform=None):
        indices = self.train_regression_idx
        dataset = WILDSSubDataset(self, indices, transform)
        return dataset           

    def get_test_regression(self, transform=None):
        indices = self.test_regression_idx
        dataset = WILDSSubDataset(self, indices, transform)
        return dataset                 