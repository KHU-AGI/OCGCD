#
# Created on March 2022
#
# Copyright (c) 2022 Meitar Ronen
#

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

def transform_embeddings(transform_type, data):
    if transform_type == "normalize":
        return torch.Tensor(Normalizer().fit_transform(data))
    elif transform_type == "min_max":
        return torch.Tensor(MinMaxScaler().fit_transform(data))
    elif transform_type == "standard":
        return torch.Tensor(StandardScaler().fit_transform(data))
    else:
        raise NotImplementedError()

class MyDataset:
    @property
    def data_dim(self):
        """
        The dimension of the loaded data
        """
        return self._data_dim

    def __init__(self, args):
        self.args = args
        self.data_dir = args.dir

    def get_train_data(self):
        raise NotImplementedError()

    def get_test_data(self):
        raise NotImplementedError()

    def get_train_loader(self):
        train_loader = torch.utils.data.DataLoader(
            self.get_train_data(),
            # batch_size=self.args.batch_size,
            batch_size=self.args.sz_batch,
            shuffle=True, #True,
            num_workers=6,
        )
        return train_loader

    def get_test_loader(self):
        test_data = self.get_test_data()
        if len(test_data) > 0:     
            return torch.utils.data.DataLoader(test_data, batch_size=self.args.batch_size, shuffle=False, num_workers=6)
        else:
            return None

    def get_loaders(self):
        return self.get_train_loader(), self.get_test_loader()


class MNIST(MyDataset):
    def __init__(self, args):
        super().__init__(args)
        self.transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self._data_dim = 28 * 28

    def get_train_data(self):
        return datasets.MNIST(self.data_dir, train=True, download=True, transform=self.transformer)

    def get_test_data(self):
        return datasets.MNIST(self.data_dir, train=False, transform=self.transformer)


class STL10(MyDataset):
    def __init__(self, args, split="train"):
        super().__init__(args)
        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self._data_dim = 96 * 96 * 3
        self.split = split

    def get_train_data(self):
        return datasets.STL10(self.data_dir, split=self.split, download=True, transform=self.transformer)

    def get_test_data(self):
        return datasets.STL10(self.data_dir, split="test", transform=self.transformer)


class TensorDatasetWrapper(TensorDataset):
    def __init__(self, data, labels):
        super().__init__(data, labels)
        self.data = data
        self.targets = labels


class CustomDataset(MyDataset):
    def __init__(self, args):
        super().__init__(args)
        self.transformer = transforms.Compose([transforms.ToTensor()])
        self._data_dim = 0
    
    def get_train_data(self):
        train_codes = torch.Tensor(torch.load(os.path.join(self.data_dir, "train_data.pt")))
        if self.args.transform_input_data:
            train_codes = transform_embeddings(self.args.transform_input_data, train_codes)
        if self.args.use_labels_for_eval:
            train_labels = torch.load(os.path.join(self.data_dir, "train_labels.pt"))
        else:
            train_labels = torch.zeros((train_codes.size()[0]))
        self._data_dim = train_codes.size()[1]
        
        train_set = TensorDatasetWrapper(train_codes, train_labels)
        del train_codes
        del train_labels
        return train_set

    def get_test_data(self):
        try:
            test_codes = torch.load(os.path.join(self.data_dir, "test_data.pt"))
            if self.args.use_labels_for_eval:
                test_labels = torch.load(os.path.join(self.data_dir, "test_labels.pt"))
            else:
                test_labels = torch.zeros((test_codes.size()[0]))
        except FileNotFoundError:
            print("Test data not found! running only with train data")
            return TensorDatasetWrapper(torch.empty(0), torch.empty(0))
        
        if self.args.transform_input_data:
            test_codes = transform_embeddings(self.args.transform_input_data, test_codes)
        test_set = TensorDatasetWrapper(test_codes, test_labels)
        del test_codes
        del test_labels
        return test_set


def merge_datasets(set_1, set_2):
    """
    Merged two TensorDatasets into one
    """
    merged = torch.utils.data.ConcatDataset([set_1, set_2])
    return merged


def generate_mock_dataset(dim, len=3, dtype=torch.float32):
    """Generates a mock TensorDataset

    Args:
        dim (tuple): shape of the sample
        len (int): number of samples. Defaults to 10.
    """
    # Make sure train and test set are of the same type
    if type(dim) == int:
        data = torch.rand((len, dim))
    else:
        data = torch.rand((len, *dim))
    data = torch.tensor(data.clone().detach(), dtype=dtype)
    return TensorDataset(data, torch.zeros(len))
