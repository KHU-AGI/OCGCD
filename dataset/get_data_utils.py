import numpy as np
from torch.utils.data import Dataset

def subsample_instances(dataset, prop_indices_to_subsample=0.8):

    np.random.seed(0)
    subsample_indices = np.random.choice(dataset.uq_idxs, replace=False,
                                         size=(int(prop_indices_to_subsample * len(dataset)),))
    # subsample_indices = np.random.choice(range(len(dataset)), replace=False,
    #                                      size=(int(prop_indices_to_subsample * len(dataset)),))

    return subsample_indices

def get_class_splits(args):
    # -------------
    # GET CLASS SPLITS
    # -------------
    if args.dataset == 'cifar10':

        args.image_size = 32
        # args.train_classes = range(5)
        # args.unlabeled_classes = range(5, 10)
        args.train_classes = range(args.num_base_classes)
        args.unlabeled_classes = range(args.num_base_classes, 10)

    elif args.dataset == 'cifar100':

        args.image_size = 32
        # args.train_classes = range(80)
        # args.unlabeled_classes = range(80, 100)
        args.train_classes = range(args.num_base_classes)
        args.unlabeled_classes = range(args.num_base_classes, 100)

    elif args.dataset == 'tinyimagenet':

        args.image_size = 64
        # args.train_classes = range(100)
        # args.unlabeled_classes = range(100, 200)
        args.train_classes = range(args.num_base_classes)
        args.unlabeled_classes = range(args.num_base_classes, 200)

    elif args.dataset == 'imagenet_100':

        args.image_size = 224
        args.train_classes = range(args.num_base_classes)
        args.unlabeled_classes = range(args.num_base_classes, 100)

    elif args.dataset == 'scars':

        args.image_size = 224

        # args.train_classes = range(98)
        # args.unlabeled_classes = range(98, 196)
        args.train_classes = range(args.num_base_classes)
        args.unlabeled_classes = range(args.num_base_classes, 196)

    elif args.dataset == 'air':
        args.image_size = 224
        # args.train_classes = range(50)
        # args.unlabeled_classes = range(50, 100)
        args.train_classes = range(args.num_base_classes)
        args.unlabeled_classes = range(args.num_base_classes, 100)

    elif args.dataset == 'cub':

        args.image_size = 224
        # if args.AND:    #* Base: 120 / Inc 60 / Unknown 20
        #     args.train_classes = range(args.num_base_classes)
        #     args.unlabeled_classes = range(args.num_base_classes, args.num_base_classes+60)
        #     args.unknown_classes = range(args.num_base_classes+60, 200)
        # else:
        #     args.train_classes = range(args.num_base_classes)
        #     args.unlabeled_classes = range(args.num_base_classes, 200)
        args.train_classes = range(args.num_base_classes)
        args.unlabeled_classes = range(args.num_base_classes, 200)
        

    else:

        raise NotImplementedError

    return args

class MergedDataset(Dataset):

    """
    Takes two datasets (labelled_dataset, unlabelled_dataset) and merges them
    Allows you to iterate over them in parallel
    """

    def __init__(self, labelled_dataset, unlabelled_dataset):

        self.labelled_dataset = labelled_dataset
        self.unlabelled_dataset = unlabelled_dataset
        self.target_transform = None

    def __getitem__(self, item):

        if item < len(self.labelled_dataset):
            img, label, uq_idx = self.labelled_dataset[item]
            labeled_or_not = 1

        else:

            img, label, uq_idx = self.unlabelled_dataset[item - len(self.labelled_dataset)]
            labeled_or_not = 0


        return img, label, uq_idx, np.array([labeled_or_not])

    def __len__(self):
        return len(self.unlabelled_dataset) + len(self.labelled_dataset)