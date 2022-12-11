import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision.datasets.utils import download_url
from wilds.datasets.wilds_dataset import WILDSDataset, WILDSSubset


class PDSDataset(Dataset):
    """
    Prototype of a PDS Dataset
    """
    @property
    def num_batches(self) -> int:
        return self._n_batches

    @property
    def num_classes(self) -> int:
        return self._n_classes

    def get_batch(self, t=None, transform=None, idx=None):
        """Should return a PDSSubDataset"""
        raise NotImplementedError

    def get_test_batch(self, t=None, transform=None):
        """Get an i.i.d. batch of the training batch with the same t.
        Should return a PDSSubDataset"""
        raise NotImplementedError    

    def get_train_regression(self, transform=None):
        """Get a train regression set.
        Should return a PDSSubDataset"""
        raise NotImplementedError

    def get_test_regression(self, transform=None):
        """Get a test regression set.
        This set should not be used for training."""
        raise NotImplementedError    


class PDSSubDataset(Dataset):
    """
    Prototype of a PDS Sub Dataset
    """
    def get_subset(self, indices=None, labeled=True):
        raise NotImplementedError

    def __add__(self, other):
        """Concat two PDSSubDataset's"""
        return super().__add__(other)


class UnlabeledPDSSubDataset(PDSSubDataset):
    def __init__(self, dataset: PDSSubDataset):
        self.dataset = dataset

    def __getitem__(self, index):
        a = self.dataset[index]
        return a[0]

    def get_subset(self, indices=None, labeled=True):
        return self.dataset.get_subset(indices, labeled=False)

    def __add__(self, other):
        return UnlabeledPDSSubDataset(self.dataset + other.dataset)

    def __len__(self):
        return len(self.dataset)


class DefaultPDSSubDataset(PDSSubDataset):
    def __init__(self, dataset: Dataset, idx_array, transform=None):
        self.dataset = dataset
        self.idx_array = idx_array
        self.transform = transform

    def __getitem__(self, index):
        i = self.idx_array[index]
        x, y = self.dataset[i]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def get_subset(self, indices=None, labeled=True):
        idx_array = self.idx_array[indices]
        res = DefaultPDSSubDataset(self.dataset, idx_array)
        if not labeled:
            res = UnlabeledPDSSubDataset(res)
        return res

    def __add__(self, other):
        idx_array = np.concatenate((self.idx_array, other.idx_array))
        return DefaultPDSSubDataset(self.dataset, idx_array)

    def __len__(self):
        return len(self.idx_array)


class TransformTensorDataset(TensorDataset, PDSSubDataset):
    def __init__(self, *tensors, transform):
        super().__init__(*tensors)
        self.transform = transform

    def __getitem__(self, index):
        a = super().__getitem__(index)
        if self.transform is not None:
            a = self.transform(a)
        return a

    def get_subset(self, indices=None, labeled=True):
        tensors = self.tensors if indices is None else [t[indices] for t in self.tensors]
        if labeled:
            return TransformTensorDataset(*tensors, transform=self.transform)
        else:
            return TransformTensorDataset(tensors[0], transform=self.transform)
    
    def __add__(self, other):
        if other is None:
            return self
        n = len(self.tensors)
        tensors = [torch.cat((self.tensors[i], other.tensors[i])) for i in range(n)]
        return TransformTensorDataset(*tensors, transform=self.transform)


class WILDSSubDataset(WILDSSubset, PDSSubDataset):
    def get_subset(self, indices=None, labeled=True):
        indices = self.indices if indices is None else self.indices[indices]
        if labeled:
            return WILDSSubDataset(self.dataset, indices, self.transform, self.do_transform_y)
        else:
            return WILDSUnlabeledSubDataset(self.dataset, indices, self.transform, self.do_transform_y)

    def __getitem__(self, idx):
        x, y, md = super().__getitem__(idx)
        return x, y

    def __add__(self, other):
        if other is None:
            return self
        indices = np.concatenate((self.indices, other.indices))
        return WILDSSubDataset(self.dataset, indices, self.transform, self.do_transform_y)


class WILDSUnlabeledSubDataset(WILDSSubDataset):
    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        return x

    def __add__(self, other):
        if other is None:
            return self
        indices = np.concatenate((self.indices, other.indices))
        return WILDSUnlabeledSubDataset(self.dataset, indices, self.transform, self.do_transform_y)


class WPDSDataset(WILDSDataset, PDSDataset):
    @property
    def num_groups(self) -> int:
        return self._n_groups

    @property
    def num_group_samples(self) -> list:
        return self._n_group_samples

    def get_batch(self, t=None, transform=None, idx=None):
        assert t is not None or idx is not None
        # Set idx from t
        if t is not None:
            assert t >= 0 and t < self._n_batches
            indices = np.concatenate([self.idx_array[i][self.idx_lut[i, :t].sum():self.idx_lut[i, :(t + 1)].sum()] for i in range(self._n_groups)])
        else:
            indices = np.concatenate([self.idx_array[i][int(l * self.num_group_samples[i] / 100):int(r * self.num_group_samples[i] / 100)] for (i, l, r) in idx])
        dataset = WILDSSubDataset(self, indices, transform)
        return dataset
        
    def get_test_batch(self, t=None, transform=None):
        assert t >= 0 and t < self._n_batches
        indices = np.concatenate([self.idx_array[i][self.idx_lut[i, :].sum() + self.test_idx_lut[i, :t].sum()
            :self.idx_lut[i, :].sum() + self.test_idx_lut[i, :(t + 1)].sum()] for i in range(self._n_groups)])
        dataset = WILDSSubDataset(self, indices, transform)
        return dataset
