# from .cub import CUBirds
import numpy as np
from .cub import CUBirds,get_cub_datasets, get_cub_datasets_AND
from .air import get_airs_datasets
from .imagenet import get_imagenet_100_datasets
from .cifar100 import get_cifar_datasets

from .mit import MITs
from .dog import Dogs
from .air import Airs

from .import utils
from .base import BaseDataset

import torch
# _type = {
#     'cub': get_cub_datasets,
#     'mit': MITs,
#     'dog': Dogs,
#     'air': Airs,
# }

def load(name, root, args, train_transform=None, test_transform=None, AND=False):
    #todo Data 불러오고 Get dataset으로 처음에 전부 만들어 버리자!!!
    # if AND:
    #     datasets = get_cub_datasets_AND(root, train_transform, test_transform, train_classes=args.train_classes, unlab_classes=args.unlabeled_classes, num_inc_sessions=args.num_inc_sessions, prop_train_labels=0.8, split_train_val=False, seed=args.seed, black_classes=args.unknown_classes)
    #     target_transform_dict = {}
    #     for i, cls in enumerate(list(args.train_classes) + list(args.unlabeled_classes) + list(args.unknown_classes)):
    #         target_transform_dict[cls] = i
    #     target_transform = lambda x: target_transform_dict[x]
    # else:
    
    if args.dataset == 'air':
        datasets = get_airs_datasets(root, train_transform, test_transform, train_classes=args.train_classes, unlab_classes=args.unlabeled_classes, num_inc_sessions=args.num_inc_sessions, prop_train_labels=args.prop_samples, split_train_val=False, seed=args.seed)
    elif args.dataset == 'cub':
        datasets = get_cub_datasets(root, train_transform, test_transform, train_classes=args.train_classes, unlab_classes=args.unlabeled_classes, num_inc_sessions=args.num_inc_sessions, prop_train_labels=args.prop_samples, split_train_val=False, seed=args.seed)
    elif args.dataset == 'cifar100':
        datasets = get_cifar_datasets(root, train_transform, test_transform, train_classes=args.train_classes, unlab_classes=args.unlabeled_classes, num_inc_sessions=args.num_inc_sessions, prop_train_labels=args.prop_samples, split_train_val=False, seed=args.seed)
    elif args.dataset == 'imagenet_100':
        datasets = get_imagenet_100_datasets(root, train_transform, test_transform, train_classes=args.train_classes, unlab_classes=args.unlabeled_classes, num_inc_sessions=args.num_inc_sessions, prop_train_labels=args.prop_samples, split_train_val=False, seed=args.seed)
    # elif args.dataset == 'imagenet_100':
    #     datasets = get_imagenet_100_datasets(root, train_transform, test_transform, train_classes=args.train_classes, unlab_classes=args.unlabeled_classes, num_inc_sessions=args.num_inc_sessions, prop_train_labels=args.prop_samples, split_train_val=False, seed=args.seed)

    if args.dataset != 'imagenet_100':
        target_transform_dict = {}
        for i, cls in enumerate(list(args.train_classes) + list(args.unlabeled_classes)):
            target_transform_dict[cls] = i
        
        target_transform = lambda x: target_transform_dict[x]
    else:
        target_transform = None
    
    
    # target_transform_dict = {}
    # for i, cls in enumerate(list(args.train_classes) + list(args.unlabeled_classes)):
    #     target_transform_dict[cls] = i
    # target_transform = lambda x: target_transform_dict[x]

    if args.dataset == 'imagenet_100':
        for dataset_name, dataset in datasets.items():
            print(f"[{dataset_name}]")
            if isinstance(dataset, list):
                for si, dt in enumerate(dataset):
                    dt.ys = dt.targets
                    print("Incremental Session:", si)
                    print('samples',dt.imgs.shape)
                    print('targets',dt.targets.shape)
                    print('classes',np.unique(dt.targets))
                    print('dataset.uq_idxs:',len(dt.uq_idxs))
                    print()
                    print("# of samples per Classes!!")
                    dt.new_inc = np.unique(dt.targets)
                    
            elif dataset is not None:
                tmp_loader = torch.utils.data.DataLoader(dataset, batch_size=args.sz_batch, shuffle=True, num_workers=args.nb_workers)    
                print('samples',dataset.imgs.shape)
                print('targets',dataset.targets.shape)
                print('classes',np.unique(dataset.targets))
                print('dataset.uq_idxs:',len(dataset.uq_idxs))
                print()
                dataset.ys = dataset.targets
                dataset.new_inc = np.unique(dataset.targets)
    else:
        for dataset_name, dataset in datasets.items():
            # print(f"[{dataset_name}]")
            if isinstance(dataset, list):
                for dt in dataset:
                    dt.target_transform = target_transform
            elif dataset is not None:
                dataset.target_transform = target_transform
            for dataset_name, dataset in datasets.items():
                if isinstance(dataset, list):
                    print(f"[{dataset_name}]")
                    for si, dt in enumerate(dataset):
                        print("Incremental Session:", si)
                        tmp_loader = torch.utils.data.DataLoader(dt, batch_size=args.sz_batch, shuffle=True, num_workers=args.nb_workers)    
                        xs=[]
                        ys=[]
                        for x,y,_ in tmp_loader:
                            if isinstance(x, list):
                                x = torch.cat(x,dim=0)
                            xs.append(x)
                            ys.append(y)
                        xs = torch.cat(xs,dim=0)
                        ys = torch.cat(ys,dim=0)
                        print('samples',xs.shape)
                        print('targets',ys.shape)
                        print('classes',torch.unique(ys))
                        print('dataset.ys:',len(dt.ys))
                        print()
                        print("# of samples per Classes!!")
                        dt.new_inc = torch.unique(ys)

                        
                elif dataset is not None:
                    tmp_loader = torch.utils.data.DataLoader(dataset, batch_size=args.sz_batch, shuffle=True, num_workers=args.nb_workers)    
                    xs=[]
                    ys=[]
                    for x,y,_ in tmp_loader:
                        if isinstance(x, list):
                            x = torch.cat(x,dim=0)
                        xs.append(x)
                        ys.append(y)
                    xs = torch.cat(xs,dim=0)
                    ys = torch.cat(ys,dim=0)
                    print(dataset_name)
                    print('samples',xs.shape)
                    print('targets',ys.shape)
                    print('classes',torch.unique(ys))
                    
                    print('dataset.ys:',len(dataset.ys))
                    print()
                    # print("# of samples per Classes!!")
                    # for c in torch.unique(ys):
                    #     mask = np.zeros(len(ys)).astype('bool')
                    #     mask[np.where(ys == c)[0]] = True
                    #     print("class:", c, "samples:", sum(mask))
                    if "whole_test_set" == dataset_name:
                        dataset.new_inc = torch.unique(ys)

    
    # print('[Train] Labelled Samples for Base Session:',len(datasets['train_labelled'].data))
    print('[Train] Labelled Samples for Base Session:',len(datasets['train_labelled']))
    print('[Train] # of incremental Sessions (Unlabelled):',len(datasets['train_inc_unlabelled']))
    
    # print('[Test] Labelled Samples for Base Session:',len(datasets['test_labelled'].data))
    print('[Test] Labelled Samples for Base Session:',len(datasets['test_labelled']))
    print('[Train] # of incremental Sessions (Unlabelled):',len(datasets['test_inc_inlabelled']))
    
    # print('\nBlack Category')
    # print("Classes:",np.unique(datasets['black_dataset'].ys[datasets['black_dataset'].uq_idxs]))
    
    if AND:
        return datasets['train_labelled'], datasets['train_inc_unlabelled'], datasets['test_labelled'], datasets['test_inc_inlabelled'], datasets['whole_test_set'], datasets['whole_training_set '] 
    else:
        return datasets['train_labelled'], datasets['train_inc_unlabelled'], datasets['test_labelled'], datasets['test_inc_inlabelled'], datasets['whole_test_set'], datasets['whole_training_set']  
    # return train_dataset, test_dataset, unlabelled_train_examples_test, datasets
    #* return _type[name](root=root, mode=mode, transform=transform)