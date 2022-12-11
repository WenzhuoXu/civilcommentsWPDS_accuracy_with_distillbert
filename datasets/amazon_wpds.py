import numpy as np
import os
import pickle

import wilds
from wilds.datasets.amazon_dataset import AmazonDataset
from wilds.common.grouper import CombinatorialGrouper

from datasets.pds_dataset import WPDSDataset, WILDSSubDataset


class AmazonWPDS(WPDSDataset, AmazonDataset):
    _idx_lut = np.array([
        [0, 500, 500, 1550, 1550, 1550, 2950, 2950, 2950, 2950, 4250, 4250, 4250, 4250, 5000, 5000, 5000, 5000],
        [50000, 4500, 4500, 3450, 3450, 3450, 2050, 2050, 2050, 2050, 750, 750, 750, 750, 0, 0, 0, 0]
    ])

    _test_idx_lut = np.array([
        [0, 100, 100, 310, 310, 310, 590, 590, 590, 590, 850, 850, 850, 850, 1000, 1000, 1000, 1000],
        [1000, 900, 900, 690, 690, 690, 410, 410, 410, 410, 150, 150, 150, 150, 0, 0, 0, 0]
    ])

    def __init__(self, *args, magic=42, **kwargs):
        kwargs['split_scheme'] = 'official' # Will change this later
        kwargs['version'] = '2.1'
        AmazonDataset.__init__(self, *args, **kwargs)
        self.year_id = 3
        assert self._metadata_fields[self.year_id] == 'year'
        self.category_id = 2
        assert self._metadata_fields[self.category_id] == 'category'
        self._eval_grouper = CombinatorialGrouper(
            dataset=self, groupby_fields=["year"]
        )
        self._split_scheme = 'time'

        self._n_batches = 18
        self._n_classes = 2
        self._n_groups = 2
        self._group_array = np.zeros((len(self),), dtype=int)
        self._group_array[self._y_array == 2] = 1
        self._group_array[self._y_array == 4] = 1
        self._y_array[self._y_array <= 2] = 0
        self._y_array[self._y_array >= 3] = 1
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
        b = self.idx_array[1][-4000:-1000]
        indices = b[self._metadata_array[b, self.category_id] < 10]
        dataset = WILDSSubDataset(self, indices, transform)
        return dataset        

    def get_test_regression(self, transform=None):
        b = self.idx_array[1][-1000:]
        indices = b[self._metadata_array[b, self.category_id] < 10]
        dataset = WILDSSubDataset(self, indices, transform)
        return dataset        