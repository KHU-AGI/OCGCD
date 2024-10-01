from .base import *
import torch
from torchvision import transforms, datasets

class Airs(BaseDataset):
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.mode = mode
        self.transform = transform

        self.path_train_o = self.root + '/train_o'
        self.path_train_n_1 = self.root + '/train_n_1'
        self.path_eval_o = self.root + '/valid_o'
        self.path_eval_n_1 = self.root + '/valid_n_1'

        if self.mode == 'train_0':
            self.classes = range(0, 80)
            self.path = self.path_train_o

        elif self.mode == 'train_1':
            # self.classes = range(0, 100)
            self.path = self.path_train_n_1

        elif self.mode == 'eval_0':
            self.classes = range(0, 80)
            self.path = self.path_eval_o

        elif self.mode == 'eval_1':
            self.classes = range(0, 100)
            self.path = self.path_eval_n_1

        BaseDataset.__init__(self, self.path, self.mode, self.transform)

        index = 0
        for i in datasets.ImageFolder(root=self.path).imgs:
            # i[1]: label, i[0]: the full path to an image
            y = i[1]
            # fn needed for removing non-images starting with `._`
            fn = os.path.split(i[0])[1]
            self.ys += [y]
            self.I += [index]
            self.im_paths.append(i[0])
            index += 1
