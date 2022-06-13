import glob
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import random
import numpy as np


class ImageDataset(Dataset):
    def __init__(self, opt, transforms_=None, mode='train'):
        self.opt = opt
        root = self.opt.dataPath
        self.mode = mode
        self.transform = transforms.Compose(transforms_)
        self.files_A = sorted(glob.glob(os.path.join(root, '%sA' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%sB' % mode) + '/*.*'))

    def __getitem__(self, index):
        if self.mode == 'train':
            imgSize = np.array(Image.open(self.files_A[index % len(self.files_A)]))[:, :, 0]

            H, W = imgSize.shape[0], imgSize.shape[1]

            w_offset = random.randint(0, max(0, W - self.opt.CropSize - 1))
            h_offset = random.randint(0, max(0, H - self.opt.CropSize - 1))

            item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))  # 3,384,384
            item_B = self.transform(Image.open(self.files_A[index % len(self.files_B)]))  # 3,384,384

            item_A = item_A[:, h_offset:h_offset + self.opt.CropSize, w_offset:w_offset + self.opt.CropSize]
            item_B = item_B[:, h_offset:h_offset + self.opt.CropSize, w_offset:w_offset + self.opt.CropSize]

        if self.mode == 'test':
            item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))  # 3,384,384
            item_B = self.transform(Image.open(self.files_A[index % len(self.files_B)]))

        return {'low': item_A, 'normal': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
