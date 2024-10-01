
from .base import *
import torch
import pandas as pd
import numpy as np


from torchvision import transforms, datasets
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from torchvision.datasets.folder import default_loader

from copy import deepcopy
import random

from .get_data_utils import subsample_instances
# from .cub_gcd import CustomCub2011
#todo data preprocessing 필요 --> AND로 해야하니까 우선 Meta로 맞추는게 좋을 듯;;


class CUBirds(Dataset):
    base_folder = 'images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, target_transform=None, loader=default_loader, download=False):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        self.loader = loader
        self.train = train
        
        self.binary = False
        self.ablation = False
        
        self.extra_trsf = False

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.uq_idxs = np.array(range(len(self)))
        # self.base_uq_idxs= np.array([])

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        
        
        self.data = data.merge(train_test_split, on='img_id')
        # print(classes.merge(train_test_split, on='img_id'))

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
            self.instance_classes = np.array(self.data['target'])
            self.ys = np.array(self.data['target']-1)
        else:
            self.data = self.data[self.data.is_training_img == 0]
            self.ys = np.array(self.data['target']-1)
            #todo im_paths / I

    def _check_integrity(self):
        self._load_metadata()

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print('integrity False:', filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)
        
        if self.transform is not None:
            if self.extra_trsf:
                img = self.extra_trsf(img)
            else:
                img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.binary:
            target = self.ys[idx]
            
            real_target = sample.target - 1
            if self.target_transform is not None:
                real_target = self.target_transform(real_target)
            return img, target, real_target, self.uq_idxs[idx]
        
        return img, target, self.uq_idxs[idx]
        # return img, target, idx


def online_subsample_dataset(dataset, idxs):
    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.data = dataset.data[mask]
    dataset.uq_idxs = dataset.uq_idxs[mask]
    dataset.ys = dataset.ys[mask]

    return dataset

def subsample_dataset(dataset, idxs):
    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.data = dataset.data[mask]
    dataset.uq_idxs = dataset.uq_idxs[mask]

    return dataset


def subsample_classes(dataset, include_classes=range(160)):

    include_classes_cub = np.array(include_classes) + 1     # CUB classes are indexed 1 --> 200 instead of 0 --> 199
    cls_idxs = [x for x, (_, r) in enumerate(dataset.data.iterrows()) if int(r['target']) in include_classes_cub]

    # TODO: For now have no target transform
    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)


    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = np.unique(train_dataset.data['target'])

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.data['target'] == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs

#!===================================================================================================================================================================================
#todo Incremental Session Update
#todo 1. Based on CGCD 2step ours (4step onn CUB): D0, D1, D2, D3, D4
#todo 2. Base Step: D0 (Labeled) / 0.8 || D0 (Unlabelled) / 0.2 --> Split into 4 sessions! == split-D0
#todo 2. Inc 1 Step: split-D0 (unlabelled) + D1 (Unlabelled) (0.8) || D1 (Unlabelled) / 0.2 --> Split into 3 sessions! == split-D1
#todo 2. Inc 2 Step: split-D0 (unlabelled) + split-D1 (Unlabelled)  + D2 (Unlabelled) (0.8) || D2 (Unlabelled) / 0.2 --> Split into 2 sessions! == split-D2
#todo 2. Inc 3 Step: split-D0 (unlabelled) + split-D1 (Unlabelled)  + split-D2 (Unlabelled) + D3 (Unlabelled) (0.8) || D3 (Unlabelled) / 0.2 --> Split into 1 sessions! == split-D3
#todo 2. Inc 4 Step: split-D0 (unlabelled) + split-D1 (Unlabelled)  + split-D2 (Unlabelled) + split-D3 (Unlabelled) + D4 (Unlabelled)

#todo 3. After GCD implementation --> TTA Applying // Base: 100 / train novel: 80 (20 x 4) / test novel: 20 (5 x 4)
#!===================================================================================================================================================================================
import copy

def get_cub_datasets(cub_root, train_transform, test_transform, train_classes, unlab_classes, num_inc_sessions, prop_train_labels=0.8,
                    split_train_val=False, seed=0):

    np.random.seed(seed)

    # Init entire training set
    whole_training_set = CUBirds(root=cub_root, transform=train_transform, train=True)
    #* print("[CUB200]:", len(whole_training_set.data))
    
    
    # Get labelled training set which has subsampled classes, then subsample some indices from that
    #* print('train classes:', train_classes)
    old_dataset_all = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    #* print("old_dataset_all",len(old_dataset_all))

    #todo 각 Class마다 Sample dataset 생성
    each_old_all_samples = [subsample_classes(deepcopy(old_dataset_all), include_classes=[targets])
                                                                for targets in train_classes]
    
    #todo 각 Class마다 0.8 비율로 할당
    each_old_labeled_slices = [subsample_instances(samples, prop_indices_to_subsample=prop_train_labels)
                                                                for samples in each_old_all_samples]
    base_sample_indices = np.concatenate(each_old_labeled_slices)
    #* print("base_sample_indices",len(base_sample_indices),np.unique(base_sample_indices))

    each_old_unlabeled_slices = [np.array(list(set(samples.uq_idxs) - set(each_old_labeled_slices[i])))
                                                                for i, samples in enumerate(each_old_all_samples)]
    each_old_unlabeled_slices = np.concatenate(each_old_unlabeled_slices)
    #* print("each_old_unlabeled_slices",len(each_old_unlabeled_slices),np.unique(each_old_unlabeled_slices))
    np.random.shuffle(each_old_unlabeled_slices)
    inc_unlabelled_base_idxs = np.split(each_old_unlabeled_slices, num_inc_sessions)
    
    train_dataset_labelled = subsample_dataset(old_dataset_all, base_sample_indices)
    
    #todo Incremental classes..
    unlabelled_classes = np.array(unlab_classes)
    assert len(unlabelled_classes) % num_inc_sessions == 0
    inc_unlabelled_classes_arr = np.split(unlabelled_classes, num_inc_sessions)
    #todo extract indices of incremental classes: Major (0.8) / Minor (0.2 & inc)
    train_inc_indices=[]
    subsample_indices_minors = []
    for inc_idx, inc_unlab_cls in enumerate(inc_unlabelled_classes_arr):
        #* print("prepare INC:",inc_idx)
        #* print("novel classes:", inc_unlab_cls)
        minor_indices=np.array([])
        inc_unlabelled = subsample_classes(deepcopy(whole_training_set), include_classes=inc_unlab_cls)
        
        #todo 각 Class마다 subdataset설정
        each_inc_all_samples = [subsample_classes(deepcopy(inc_unlabelled), include_classes=[targets])
                                                                for targets in inc_unlab_cls]

        #todo 각 Class마다 0.8 비율로 할당
        if num_inc_sessions-(1+inc_idx) != 0:
            #todo each_inc_all_samples --> 클래스마다의 uq_idxs, data 확보: [C1, C2, C3, ... ,CN]
            subsample_indices_major_slices = [subsample_instances(samples, prop_indices_to_subsample=prop_train_labels)
                                                                            for samples in each_inc_all_samples]
            subsample_indices_major = np.concatenate(subsample_indices_major_slices)
            
            sub_sample_indices_minor = np.array(list(set(inc_unlabelled.uq_idxs) - set(subsample_indices_major)))
            #* print("sub_sample_indices_minor", len(sub_sample_indices_minor))
            subsample_indices_minors.append(list(reversed(np.array_split(sub_sample_indices_minor, num_inc_sessions-(1+inc_idx)))))
            #! minor list 관리 필요 - 이전 세션 전부 들어가야함                
            if inc_idx == 0:    #* no inc minor // only base minor
                train_inc_indices.append(np.concatenate((inc_unlabelled_base_idxs[inc_idx], subsample_indices_major)))
            else:
                for midx, minor in enumerate(subsample_indices_minors[:-1]):
                    minor_idxs = minor.pop(-1)
                    minor_indices = np.concatenate((minor_indices, minor_idxs)).astype(int)
                
                train_inc_indices.append(np.concatenate((inc_unlabelled_base_idxs[inc_idx], subsample_indices_major, minor_indices)))
        else:
            subsample_indices_major = subsample_instances(inc_unlabelled, prop_indices_to_subsample=1.)
            if len(subsample_indices_minors) == 0:
                train_inc_indices.append(np.concatenate((inc_unlabelled_base_idxs[inc_idx], subsample_indices_major)))
            else:
                for midx, minor in enumerate(subsample_indices_minors):
                    minor_idxs = minor.pop(-1)
                    # print("minor_idxs:", minor_idxs)
                    minor_indices = np.concatenate((minor_indices, minor_idxs)).astype(int)
                train_inc_indices.append(np.concatenate((inc_unlabelled_base_idxs[inc_idx], subsample_indices_major, minor_indices)))
    
    train_inc_datasets_unlabelled = []
    for sess_idx, inc_indices in zip(range(num_inc_sessions), train_inc_indices):
        train_inc_datasets_unlabelled.append(subsample_dataset(deepcopy(whole_training_set), inc_indices))
    #todo ============================================================================
    whole_test_set = CUBirds(root=cub_root, transform=test_transform, train=False)
    
    test_dataset_labelled = subsample_classes(deepcopy(whole_test_set), include_classes=train_classes)
    testsample_indices = subsample_instances(test_dataset_labelled, prop_indices_to_subsample=1.)
    test_dataset_labelled = subsample_dataset(test_dataset_labelled, testsample_indices)
    #todo Prepare Test dataset each sessions including Base
    #! instance_classes
    test_inc_datasets = []
    aug_classes = list(train_classes)
    for inc_test_classes in inc_unlabelled_classes_arr:
        aug_classes = list(set(aug_classes+list(inc_test_classes)))
        test_inc_dataset_unlabelled = subsample_classes(deepcopy(whole_test_set), include_classes=aug_classes)
        testsample_indices = subsample_instances(test_inc_dataset_unlabelled, prop_indices_to_subsample=1.)
        test_inc_datasets.append(subsample_dataset(test_inc_dataset_unlabelled, testsample_indices))
    #todo ============================================================================

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_inc_unlabelled': train_inc_datasets_unlabelled,
        'test_labelled': test_dataset_labelled,
        'test_inc_inlabelled': test_inc_datasets,
        'whole_test_set': whole_test_set,
        'whole_training_set':whole_training_set
    }

    return all_datasets


def get_cub_datasets_AND(cub_root, train_transform, test_transform, train_classes, unlab_classes, num_inc_sessions, prop_train_labels=0.8,
                    split_train_val=False, seed=0, black_classes=None):

    np.random.seed(seed)

    # Init entire training set
    whole_training_set = CUBirds(root=cub_root, transform=train_transform, train=True)
    #* print("[CUB200]:", len(whole_training_set.data))
    
    
    # Get labelled training set which has subsampled classes, then subsample some indices from that
    #* print('train classes:', train_classes)
    old_dataset_all = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    #* print("old_dataset_all",len(old_dataset_all))

    #todo 각 Class마다 Sample dataset 생성
    each_old_all_samples = [subsample_classes(deepcopy(old_dataset_all), include_classes=[targets])
                                                                for targets in train_classes]
    
    #todo 각 Class마다 0.8 비율로 할당
    each_old_labeled_slices = [subsample_instances(samples, prop_indices_to_subsample=prop_train_labels)
                                                                for samples in each_old_all_samples]
    base_sample_indices = np.concatenate(each_old_labeled_slices)
    #* print("base_sample_indices",len(base_sample_indices),np.unique(base_sample_indices))

    each_old_unlabeled_slices = [np.array(list(set(samples.uq_idxs) - set(each_old_labeled_slices[i])))
                                                                for i, samples in enumerate(each_old_all_samples)]
    each_old_unlabeled_slices = np.concatenate(each_old_unlabeled_slices)
    #* print("each_old_unlabeled_slices",len(each_old_unlabeled_slices),np.unique(each_old_unlabeled_slices))
    np.random.shuffle(each_old_unlabeled_slices)
    inc_unlabelled_base_idxs = np.split(each_old_unlabeled_slices, num_inc_sessions)
    
    train_dataset_labelled = subsample_dataset(old_dataset_all, base_sample_indices)
    
    #todo Incremental classes..
    unlabelled_classes = np.array(unlab_classes)
    assert len(unlabelled_classes) % num_inc_sessions == 0
    inc_unlabelled_classes_arr = np.split(unlabelled_classes, num_inc_sessions)
    #todo extract indices of incremental classes: Major (0.8) / Minor (0.2 & inc)
    train_inc_indices=[]
    subsample_indices_minors = []
    for inc_idx, inc_unlab_cls in enumerate(inc_unlabelled_classes_arr):
        #* print("prepare INC:",inc_idx)
        #* print("novel classes:", inc_unlab_cls)
        minor_indices=np.array([])
        inc_unlabelled = subsample_classes(deepcopy(whole_training_set), include_classes=inc_unlab_cls)
        
        #todo 각 Class마다 subdataset설정
        each_inc_all_samples = [subsample_classes(deepcopy(inc_unlabelled), include_classes=[targets])
                                                                for targets in inc_unlab_cls]

        #todo 각 Class마다 0.8 비율로 할당
        if num_inc_sessions-(1+inc_idx) != 0:
            #todo each_inc_all_samples --> 클래스마다의 uq_idxs, data 확보: [C1, C2, C3, ... ,CN]
            subsample_indices_major_slices = [subsample_instances(samples, prop_indices_to_subsample=prop_train_labels)
                                                                            for samples in each_inc_all_samples]
            subsample_indices_major = np.concatenate(subsample_indices_major_slices)
            
            sub_sample_indices_minor = np.array(list(set(inc_unlabelled.uq_idxs) - set(subsample_indices_major)))
            #* print("sub_sample_indices_minor", len(sub_sample_indices_minor))
            subsample_indices_minors.append(list(reversed(np.array_split(sub_sample_indices_minor, num_inc_sessions-(1+inc_idx)))))
            #! minor list 관리 필요 - 이전 세션 전부 들어가야함                
            if inc_idx == 0:    #* no inc minor // only base minor
                train_inc_indices.append(np.concatenate((inc_unlabelled_base_idxs[inc_idx], subsample_indices_major)))
            else:
                for midx, minor in enumerate(subsample_indices_minors[:-1]):
                    minor_idxs = minor.pop(-1)
                    minor_indices = np.concatenate((minor_indices, minor_idxs)).astype(int)
                
                train_inc_indices.append(np.concatenate((inc_unlabelled_base_idxs[inc_idx], subsample_indices_major, minor_indices)))
        else:
            subsample_indices_major = subsample_instances(inc_unlabelled, prop_indices_to_subsample=1.)
            if len(subsample_indices_minors) == 0:
                train_inc_indices.append(np.concatenate((inc_unlabelled_base_idxs[inc_idx], subsample_indices_major)))
            else:
                for midx, minor in enumerate(subsample_indices_minors):
                    minor_idxs = minor.pop(-1)
                    # print("minor_idxs:", minor_idxs)
                    minor_indices = np.concatenate((minor_indices, minor_idxs)).astype(int)
                train_inc_indices.append(np.concatenate((inc_unlabelled_base_idxs[inc_idx], subsample_indices_major, minor_indices)))
    
    train_inc_datasets_unlabelled = []
    for sess_idx, inc_indices in zip(range(num_inc_sessions), train_inc_indices):
        train_inc_datasets_unlabelled.append(subsample_dataset(deepcopy(whole_training_set), inc_indices))
    #! Black Category
    #* black_datasets = subsample_classes(deepcopy(whole_training_set), include_classes=black_classes)
    # black_indices = subsample_instances(black_datasets, prop_indices_to_subsample=1.)
    # black_datasets = subsample_dataset(black_datasets, black_indices)
    #todo ============================================================================
    whole_test_set = CUBirds(root=cub_root, transform=test_transform, train=False)
    
    test_dataset_labelled = subsample_classes(deepcopy(whole_test_set), include_classes=train_classes)
    testsample_indices = subsample_instances(test_dataset_labelled, prop_indices_to_subsample=1.)
    test_dataset_labelled = subsample_dataset(test_dataset_labelled, testsample_indices)
    #todo Prepare Test dataset each sessions including Base
    #! instance_classes
    test_inc_datasets = []
    aug_classes = list(train_classes)
    for inc_test_classes in inc_unlabelled_classes_arr:
        aug_classes = list(set(aug_classes+list(inc_test_classes)))
        test_inc_dataset_unlabelled = subsample_classes(deepcopy(whole_test_set), include_classes=aug_classes)
        testsample_indices = subsample_instances(test_inc_dataset_unlabelled, prop_indices_to_subsample=1.)
        test_inc_datasets.append(subsample_dataset(test_inc_dataset_unlabelled, testsample_indices))
    #todo ============================================================================

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_inc_unlabelled': train_inc_datasets_unlabelled,
        'test_labelled': test_dataset_labelled,
        'test_inc_inlabelled': test_inc_datasets,
        # 'black_dataset':black_datasets,
        'whole_test_set': whole_test_set,
        'whole_training_set':whole_training_set
    }

    return all_datasets



#todo TTA How to apply?? Refer. "TENT & COTTA"
# def get_cub_datasets_AND(cub_root, train_transform, test_transform, train_classes, unlab_classes, num_inc_sessions, prop_train_labels=0.8,
#                     split_train_val=False, seed=0):

#     np.random.seed(seed)

#     # Init entire training set
#     whole_training_set = CUBirds(root=cub_root, transform=train_transform, train=True)
#     print("[CUB200]:", len(whole_training_set.data))
    
    
#     # Get labelled training set which has subsampled classes, then subsample some indices from that
#     train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
#     base_sample_idxs = copy.deepcopy(train_dataset_labelled.uq_idxs)
    
#     subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
#     train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)
    
#     #todo Base classes samples for the incremental sessions..
#     inc_unlabelled_base_idxs = np.split(np.array(list(set(base_sample_idxs) - set(train_dataset_labelled.uq_idxs))), num_inc_sessions)
    
    
#     #todo Incremental classes..
#     unlabelled_classes = np.array(unlab_classes)
#     assert len(unlabelled_classes) % num_inc_sessions == 0
#     inc_unlabelled_classes_arr = np.split(unlabelled_classes, num_inc_sessions)
#     #todo extract indices of incremental classes: Major (0.8) / Minor (0.2 & inc)
#     train_inc_indices=[]
#     subsample_indices_minors = []
#     for inc_idx, inc_unlab_cls in enumerate(inc_unlabelled_classes_arr):
#         print("prepare INC:",inc_idx)
#         print("novel classes:", inc_unlab_cls)
#         # print()
#         # print("Sess{} subsample_indices_minors".format(inc_idx))
#         # print(subsample_indices_minors)
#         # print()
#         # subsample_base_minor = inc_unlabelled_base_idxs[inc_idx]
#         minor_indices=np.array([])
        
#         inc_unlabelled = subsample_classes(deepcopy(whole_training_set), include_classes=inc_unlab_cls)
#         inc_sample_idxs = inc_unlabelled.uq_idxs
#         # if num_inc_sessions-(1+inc_idx) == 0:
#             # subsample_indices_major = subsample_instances(inc_unlabelled, prop_indices_to_subsample=1.)
#             # train_inc_indices.append(np.concatenate((inc_unlabelled_base_idxs[inc_idx], inc_unlabelled.uq_idxs[subsample_indices_major])))
#         # print("inc_idx / minor list shpae:", inc_idx, len(np.array(subsample_indices_minors)))
#         if num_inc_sessions-(1+inc_idx) != 0:
#             subsample_indices_major = subsample_instances(inc_unlabelled, prop_indices_to_subsample=prop_train_labels)
#             subsample_indices_minors.append(list(reversed(np.split(np.array(list(set(inc_sample_idxs) - set(inc_unlabelled.uq_idxs[subsample_indices_major]))), num_inc_sessions-(1+inc_idx)))))
#             #! minor list 관리 필요 - 이전 세션 전부 들어가야함                
#             if inc_idx == 0:    #* no inc minor // only base minor
#                 train_inc_indices.append(np.concatenate((inc_unlabelled_base_idxs[inc_idx], inc_unlabelled.uq_idxs[subsample_indices_major])))
#             else:
#                 for midx, minor in enumerate(subsample_indices_minors[:-1]):
#                     # print("inc_idx / midx", inc_idx, midx)
#                     minor_idxs = minor.pop(-1)
#                     # print()
#                     # print('minor:', minor_idxs)
#                     # print()
#                     minor_indices = np.concatenate((minor_indices, minor_idxs)).astype(int)
#                     # print('minor_indices:', minor_indices)
#                     # minor_indices.append(minor.pop(-1))
#                     # minor_indices = np.concatenate(minor_indices)
                
#                 train_inc_indices.append(np.concatenate((inc_unlabelled_base_idxs[inc_idx], inc_unlabelled.uq_idxs[subsample_indices_major], minor_indices)))
#         else:
#             subsample_indices_major = subsample_instances(inc_unlabelled, prop_indices_to_subsample=1.)
#             if len(subsample_indices_minors) == 0:
#                 train_inc_indices.append(np.concatenate((inc_unlabelled_base_idxs[inc_idx], inc_unlabelled.uq_idxs[subsample_indices_major])))
#             else:
#                 for midx, minor in enumerate(subsample_indices_minors):
#                     minor_idxs = minor.pop(-1)
#                     # print("minor_idxs:", minor_idxs)
#                     minor_indices = np.concatenate((minor_indices, minor_idxs)).astype(int)
#                 train_inc_indices.append(np.concatenate((inc_unlabelled_base_idxs[inc_idx], inc_unlabelled.uq_idxs[subsample_indices_major], minor_indices)))
    
#     train_inc_datasets_unlabelled = []
#     for sess_idx, inc_indices in zip(range(num_inc_sessions), train_inc_indices):
#         # print("sess_idx:",sess_idx,inc_indices)
#         train_inc_datasets_unlabelled.append(subsample_dataset(deepcopy(whole_training_set), inc_indices))
#     #todo ============================================================================
#     whole_test_set = CUBirds(root=cub_root, transform=test_transform, train=False)
    
#     test_dataset_labelled = subsample_classes(deepcopy(whole_test_set), include_classes=train_classes)
#     testsample_indices = subsample_instances(test_dataset_labelled, prop_indices_to_subsample=1.)
#     test_dataset_labelled = subsample_dataset(test_dataset_labelled, testsample_indices)
#     #todo Prepare Test dataset each sessions including Base
#     #! instance_classes
#     test_inc_datasets = []
#     aug_classes = list(train_classes)
#     for inc_test_classes in inc_unlabelled_classes_arr:
#         aug_classes = list(set(aug_classes+list(inc_test_classes)))
#         test_inc_dataset_unlabelled = subsample_classes(deepcopy(whole_test_set), include_classes=aug_classes)
#         testsample_indices = subsample_instances(test_inc_dataset_unlabelled, prop_indices_to_subsample=1.)
#         test_inc_datasets.append(subsample_dataset(test_inc_dataset_unlabelled, testsample_indices))
#     #todo ============================================================================

#     all_datasets = {
#         'train_labelled': train_dataset_labelled,
#         'train_inc_unlabelled': train_inc_datasets_unlabelled,
#         'test_labelled': test_dataset_labelled,
#         'test_inc_inlabelled': test_inc_datasets,
#         'whole_test_set': whole_test_set
#     }

#     return all_datasets


# class CUBirds(BaseDataset):
#     def __init__(self, root, mode, transform=None):
#         self.root = root
#         self.mode = mode
#         self.transform = transform

#         # self.path_train_o = self.root + '/train_o'
#         # self.path_train_n_1 = self.root + '/train_n_1'
#         # self.path_eval_o = self.root + '/valid_o'
#         # self.path_eval_n_1 = self.root + '/valid_n_1'

#         if self.mode == 'train_0':
#             self.classes = range(0, 160)
#             self.path = self.path_train_o

#         elif self.mode == 'train_1':
#             # self.classes = range(0, 200)
#             self.path = self.path_train_n_1

#         elif self.mode == 'eval_0':
#             self.classes = range(0, 160)
#             self.path = self.path_eval_o

#         elif self.mode == 'eval_1':
#             self.classes = range(0, 200)
#             self.path = self.path_eval_n_1

#         BaseDataset.__init__(self, self.path, self.mode, self.transform)

#         index = 0
#         for i in datasets.ImageFolder(root=self.path).imgs:
#             # i[1]: label, i[0]: the full path to an image
#             y = i[1]
#             # fn needed for removing non-images starting with `._`
#             fn = os.path.split(i[0])[1]
#             self.ys += [y]
#             self.I += [index]
#             self.im_paths.append(i[0])
#             index += 1