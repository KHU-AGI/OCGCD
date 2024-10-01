import torchvision
import numpy as np

import os

from copy import deepcopy
# from data.data_utils import subsample_instances
from .get_data_utils import subsample_instances
# from config import imagenet_root


class ImageNetBase(torchvision.datasets.ImageFolder):

    def __init__(self, root, transform):

        super(ImageNetBase, self).__init__(root, transform)

        self.uq_idxs = np.array(range(len(self)))
        
        self.imgs = np.array(self.imgs)
        self.samples = np.array(self.samples)
        self.targets = np.array(self.targets)
        self.ablation = False
        self.binary = False

    def __getitem__(self, item):

        # img, label = super().__getitem__(item)
        img, _ = super().__getitem__(item)
        label = self.targets[item]
        
        # print('img:', img.shape)
        # print('label:',label, type(label))
        uq_idx = self.uq_idxs[item]
        
        if self.binary:
            target = self.ys[item]
            if self.ablation:
                real_target = self.targets[item]
                # if self.target_transform is not None:
                    # real_target = self.target_transform(real_target)
                return img, target, real_target, self.uq_idxs[item]

        return img, label.astype(int), uq_idx


def subsample_dataset(dataset, idxs):
    # idxs = np.array(idxs).astype(int)
    # imgs_ = []
    # for i in idxs:
    #     imgs_.append(dataset.imgs[int(i)])
    # dataset.imgs = imgs_

    # samples_ = []
    # for i in idxs:
    #     samples_.append(dataset.samples[int(i)])
    # dataset.samples = samples_
    
    # dataset.imgs = [x for i, x in enumerate(dataset.imgs) if i in idxs]
    # dataset.samples = [x for i, x in enumerate(dataset.samples) if i in idxs]
    # dataset.targets = np.array(dataset.targets)[idxs].tolist()
    # dataset.uq_idxs = dataset.uq_idxs[idxs.astype(int)]
    
    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    # dataset.data = dataset.data[mask]
    # dataset.uq_idxs = dataset.uq_idxs[mask]
    
    dataset.imgs = dataset.imgs[mask]
    dataset.samples = dataset.samples[mask]
    dataset.targets = np.array(dataset.targets[mask]).astype(int)
    dataset.uq_idxs = dataset.uq_idxs[mask]
    
    return dataset

def online_subsample_dataset(dataset, idxs):
    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.imgs = dataset.imgs[mask]
    dataset.samples = dataset.samples[mask]
    dataset.uq_idxs = dataset.uq_idxs[mask]
    dataset.targets = np.array(dataset.targets[mask]).astype(int)
    dataset.ys = dataset.ys[mask]

    return dataset

def subsample_classes(dataset, include_classes=list(range(1000))):
    include_classes_in = np.array(include_classes)
    
    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes_in]
    # target_xform_dict = {}
    # for i, k in enumerate(include_classes):
        # target_xform_dict[k] = i
    dataset = subsample_dataset(dataset, cls_idxs)
    # dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = list(set(train_dataset.targets))

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(np.array(train_dataset.targets) == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_equal_len_datasets(dataset1, dataset2):
    """
    Make two datasets the same length
    """

    if len(dataset1) > len(dataset2):

        rand_idxs = np.random.choice(range(len(dataset1)), size=(len(dataset2, )))
        subsample_dataset(dataset1, rand_idxs)

    elif len(dataset2) > len(dataset1):

        rand_idxs = np.random.choice(range(len(dataset2)), size=(len(dataset1, )))
        subsample_dataset(dataset2, rand_idxs)

    return dataset1, dataset2

def get_imagenet_100_datasets(air_root, train_transform, test_transform, train_classes, unlab_classes, num_inc_sessions, prop_train_labels=0.8,
                    split_train_val=False, seed=0):
# def get_imagenet_100_datasets(train_transform, test_transform, train_classes=range(80),
                        #    prop_train_labels=0.8, split_train_val=False, seed=0):

    np.random.seed(seed)

    # Subsample imagenet dataset initially to include 100 classes
    # subsampled_100_classes = np.random.choice(range(1000), size=(100,), replace=False)
    subsampled_100_classes = np.arange(100)
    subsampled_100_classes = np.sort(subsampled_100_classes)
    print(f'Constructing ImageNet-100 dataset from the following classes: {subsampled_100_classes.tolist()}')
    cls_map = {i: j for i, j in zip(subsampled_100_classes, range(100))}
    # print('cls_map:',cls_map)

    # Init entire training set
    # imagenet_training_set = ImageNetBase(root=os.path.join(air_root, 'train'), transform=train_transform)
    imagenet_training_set = ImageNetBase(root=air_root+'/train', transform=train_transform)
    whole_training_set = subsample_classes(imagenet_training_set, include_classes=subsampled_100_classes)

    # print("[subsample_classes] whole training set.targets", np.unique(whole_training_set.targets))
    # print("[subsample_classes] whole training set.imgs", np.array(whole_training_set.imgs).shape)
    
    # train_classes = [v for k, v in cls_map.items() if int(k) in subsampled_100_classes.tolist()]
    train_classes = [v for k, v in cls_map.items() if int(k) in train_classes]
    # print('train_classes:',train_classes)
    
    # Reset dataset
    # print("[s[1] for s in whole_training_set.samples]", np.unique([s[1] for s in whole_training_set.samples]))
    # whole_training_set.samples = [(s[0], cls_map[int(s[1])]) for s in whole_training_set.samples]
    # whole_training_set.targets = [s[1] for s in whole_training_set.samples]
    # whole_training_set.ys = [s[1] for s in whole_training_set.samples]
    # whole_training_set.uq_idxs = np.array(range(len(whole_training_set)))
    whole_training_set.target_transform = None
    # print("[s[1] for s in whole_training_set.samples]", np.unique([s[1] for s in whole_training_set.samples]))
    # print('Reset-whole_training_set.targets', np.unique(whole_training_set.targets))
    
    
    old_dataset_all = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    # print("[imagenet old_dataset_all]",len(old_dataset_all), np.unique(old_dataset_all.targets))

    #todo 각 Class마다 Sample dataset 생성
    each_old_all_samples = [subsample_classes(deepcopy(old_dataset_all), include_classes=[targets])
                                                                for targets in train_classes]
    # print("[Aircraft old_dataset_all]",len(old_dataset_all))
    #todo 각 Class마다 0.8 비율로 할당
    each_old_labeled_slices = [subsample_instances(samples, prop_indices_to_subsample=prop_train_labels)
                                                                for samples in each_old_all_samples]
    base_sample_indices = np.concatenate(each_old_labeled_slices)
    # print("base_sample_indices",base_sample_indices, len(base_sample_indices))

    each_old_unlabeled_slices = [np.array(list(set(samples.uq_idxs) - set(each_old_labeled_slices[i])))
                                                                for i, samples in enumerate(each_old_all_samples)]
    # print('each_old_unlabeled_slices',each_old_unlabeled_slices)
    
    each_old_unlabeled_slices = np.concatenate(each_old_unlabeled_slices)
    #* print("each_old_unlabeled_slices",len(each_old_unlabeled_slices),np.unique(each_old_unlabeled_slices))
    np.random.shuffle(each_old_unlabeled_slices)
    inc_unlabelled_base_idxs = np.split(each_old_unlabeled_slices, num_inc_sessions)
    # print("[Aircraft old_dataset_all]", old_dataset_all.uq_idxs, len(old_dataset_all.uq_idxs))
    # print("[Aircraft base_sample_indices]", base_sample_indices, len(base_sample_indices))

    # train_dataset_labelled = subsample_dataset(old_dataset_all, base_sample_indices)
    train_dataset_labelled = subsample_dataset(deepcopy(whole_training_set), base_sample_indices)
    # print("[get_imagenet-100] train_dataset_labelled.targets", np.unique(train_dataset_labelled.targets))
    # print("[get_imagenet-100] train_dataset_labelled.imgs", np.array(train_dataset_labelled.imgs).shape)
    
    #todo Incremental classes..
    unlabelled_classes = np.array(unlab_classes)
    assert len(unlabelled_classes) % num_inc_sessions == 0
    inc_unlabelled_classes_arr = np.split(unlabelled_classes, num_inc_sessions)
    #todo extract indices of incremental classes: Major (0.8) / Minor (0.2 & inc)
    train_inc_indices=[]
    subsample_indices_minors = []
    for inc_idx, inc_unlab_cls_ in enumerate(inc_unlabelled_classes_arr):
        #* print("prepare INC:",inc_idx)
        #* print("novel classes:", inc_unlab_cls)
        minor_indices=np.array([])
        
        inc_unlab_cls = [k for k, v in cls_map.items() if v in inc_unlab_cls_]
        # print('inc_unlab_cls:',inc_unlab_cls)
        
        # inc_unlabelled = subsample_classes(deepcopy(whole_training_set), include_classes=inc_unlab_cls)
        # print("[imagenet whole_training_set]",len(whole_training_set), np.unique(whole_training_set.targets))
        inc_unlabelled = subsample_classes(deepcopy(whole_training_set), include_classes=inc_unlab_cls)
        
        # print("[imagenet inc_unlabelled]",len(inc_unlabelled), np.unique(inc_unlabelled.targets))
        #todo 각 Class마다 subdataset설정
        # cls_map[s[1]]
        # each_inc_all_samples = [subsample_classes(deepcopy(inc_unlabelled), include_classes=[targets])for targets in inc_unlab_cls]
        each_inc_all_samples = [subsample_classes(deepcopy(inc_unlabelled), include_classes=[cls_map[targets]])for targets in inc_unlab_cls]
        

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
        tmp_inc_dt = subsample_dataset(deepcopy(whole_training_set), inc_indices)
        train_inc_datasets_unlabelled.append(tmp_inc_dt)
        
        # print(f"[get_imagenet-100 Inc {sess_idx}] tmp_inc_dt.targets", np.unique(tmp_inc_dt.targets))
        # print(f"[get_imagenet-100 Inc {sess_idx}] tmp_inc_dt.imgs", np.array(tmp_inc_dt.imgs).shape)
    #todo ============================================================================
    imagenet_testing_set = ImageNetBase(root=air_root+'/val', transform=test_transform)
    whole_test_set = subsample_classes(imagenet_testing_set, include_classes=subsampled_100_classes)
    
    
    # print("[imagenet whole_test_set]",len(whole_test_set), np.unique(whole_test_set.targets))
    
    # whole_test_set.samples = [(s[0], cls_map[s[1]]) for s in whole_test_set.samples]
    # whole_test_set.targets = [s[1] for s in whole_test_set.samples]
    # whole_test_set.ys = [s[1] for s in whole_test_set.samples]
    # whole_test_set.uq_idxs = np.array(range(len(whole_test_set)))
    whole_test_set.target_transform = None
    
    test_dataset_labelled = subsample_classes(deepcopy(whole_test_set), include_classes=train_classes)
    testsample_indices = subsample_instances(test_dataset_labelled, prop_indices_to_subsample=1.)
    test_dataset_labelled = subsample_dataset(deepcopy(whole_test_set), testsample_indices)
    # print(f"[get_imagenet-100] test_dataset_labelled.targets", np.unique(test_dataset_labelled.targets))
    # print(f"[get_imagenet-100] test_dataset_labelled.imgs", np.array(test_dataset_labelled.imgs).shape)
    #todo Prepare Test dataset each sessions including Base
    #! instance_classes
    test_inc_datasets = []
    aug_classes = list(train_classes)
    for inc_test_classes in inc_unlabelled_classes_arr:
        aug_classes = list(set(aug_classes+list(inc_test_classes)))
        test_inc_dataset_unlabelled = subsample_classes(deepcopy(whole_test_set), include_classes=aug_classes)
        testsample_indices = subsample_instances(test_inc_dataset_unlabelled, prop_indices_to_subsample=1.)
        inc_test_dt = subsample_dataset(deepcopy(whole_test_set), testsample_indices)
        
        # print(f"[get_imagenet-100] inc_test_dt.targets", np.unique(inc_test_dt.targets))
        # print(f"[get_imagenet-100] inc_test_dt.imgs", np.array(inc_test_dt.imgs).shape)
        
        test_inc_datasets.append(inc_test_dt)
    #todo ============================================================================

    # 'train_labelled': train_dataset_labelled
    # train_dataset_labelled.targets = list(train_dataset_labelled.targets) 
    
    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_inc_unlabelled': train_inc_datasets_unlabelled,
        'test_labelled': test_dataset_labelled,
        'test_inc_inlabelled': test_inc_datasets,
        'whole_test_set': whole_test_set,
        'whole_training_set':whole_training_set
    }

    return all_datasets
    

    # # Get labelled training set which has subsampled classes, then subsample some indices from that
    # train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    # subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    # train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # # Split into training and validation sets
    # train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    # train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    # val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    # val_dataset_labelled_split.transform = test_transform

    # # Get unlabelled data
    # unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    # train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

    # # Get test set for all classes
    # test_dataset = ImageNetBase(root=os.path.join(imagenet_root, 'val'), transform=test_transform)
    # test_dataset = subsample_classes(test_dataset, include_classes=subsampled_100_classes)

    # # Reset test set
    # test_dataset.samples = [(s[0], cls_map[s[1]]) for s in test_dataset.samples]
    # test_dataset.targets = [s[1] for s in test_dataset.samples]
    # test_dataset.uq_idxs = np.array(range(len(test_dataset)))
    # test_dataset.target_transform = None

    # # Either split train into train and val or use test set as val
    # train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    # val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    # all_datasets = {
    #     'train_labelled': train_dataset_labelled,
    #     'train_unlabelled': train_dataset_unlabelled,
    #     'val': val_dataset_labelled,
    #     'test': test_dataset,
    # }

    # return all_datasets


if __name__ == '__main__':

    x = get_imagenet_100_datasets(None, None, split_train_val=False,
                               train_classes=range(50), prop_train_labels=0.5)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    print(f'Num Labelled Classes: {len(set(x["train_labelled"].targets))}')
    print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].targets))}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')