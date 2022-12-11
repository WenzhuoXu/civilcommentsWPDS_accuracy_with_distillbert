import numpy as np
import os
import pickle

import wilds
from wilds.datasets.civilcomments_dataset import CivilCommentsDataset
from datasets.pds_dataset import WPDSDataset, WILDSSubDataset


class CivilCommentsWPDS(WPDSDataset, CivilCommentsDataset):
    _idx_lut = np.array([
        [50000,9000,9000,7500,7500,6000,6000,4000,4000,4000,4000,4000,4000,4000,4000,4000],
        [0,1000,1000,2500,2500,4000,4000,6000,3000,3000,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,3000,3000,6000,6000,3000,3000,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,3000,3000,6000,6000]
    ])

    _test_idx_lut = np.array([
        [5000,900,900,750,750,600,600,400,400,400,400,400,400,400,400,400],
        [0,100,100,250,250,400,400,600,300,300,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,300,300,600,600,300,300,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,300,300,600,600]
    ])

    def __init__(self, *args, magic=42, **kwargs):
        kwargs['version'] = '1.0'
        CivilCommentsDataset.__init__(self, *args, **kwargs)

        self._n_batches = 16
        self._n_classes = 2
        self._n_groups = 4
        self._group_array = np.zeros((len(self),), dtype=int)
        self._group_array[(self._metadata_array[:, 0] == 1) | (self._metadata_array[:, 1] == 1) | (self._metadata_array[:, 2] == 1)] = 3
        self._group_array[(self._metadata_array[:, 3] == 1) | (self._metadata_array[:, 4] == 1) | (self._metadata_array[:, 5] == 1)] = 2
        self._group_array[(self._metadata_array[:, 6] == 1) | (self._metadata_array[:, 7] == 1)] = 1
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

    def get_train_regression(self, transform=None):
        # Regression set: identity_attack and sexual_explicit
        idx = []
        for i in range(self.num_groups):
            a = self.idx_array[i][-7000:-1000]
            b = a[(self._metadata_array[a, 13] == 1) | (self._metadata_array[a, 14] == 1)]
            idx.append(b)
        indices = np.concatenate(idx)
        dataset = WILDSSubDataset(self, indices, transform)
        return dataset        

    def get_test_regression(self, transform=None):
        idx = []
        for i in range(self.num_groups):
            a = self.idx_array[i][-1000:]
            b = a[(self._metadata_array[a, 13] == 1) | (self._metadata_array[a, 14] == 1)]
            idx.append(b)
        indices = np.concatenate(idx)
        dataset = WILDSSubDataset(self, indices, transform)
        return dataset      