import glob
import random
import os
import numpy as np
import torch
import copy
import cv2

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)

        self.l = np.load(os.path.join(root,"l/gray_scale.npy"))
        ab1 = np.load(os.path.join(root,"ab", "ab1.npy"))
        ab2 = np.load(os.path.join(root, "ab", "ab2.npy"))
        ab3 = np.load(os.path.join(root,"ab", "ab3.npy"))
        self.ab = np.concatenate([ab1, ab2, ab3], axis=0)
        self.imgs = np.concatenate((np.expand_dims(self.l,axis=-1),self.ab), axis=3)[:3000]
    def __getitem__(self, index):

        img_x = self.imgs[index % len(self.imgs)]
        img_rgb = cv2.cvtColor(img_x,cv2.COLOR_Lab2RGB)
        img_c = Image.fromarray(img_rgb,'RGB')
        img_gray = self.l[index % len(self.l)]
        img_g = np.dstack((img_gray, img_gray, img_gray))
        img_g = Image.fromarray(img_g,'RGB')
        """ w, h = img_x.size
        img_A = img_x.crop((0, 0, w / 2, h))
        img_B = img_x.crop((w / 2, 0, w, h))

        if np.random.random() < 0.5: 
            img_A = Image.fromarray(np.array(img_x)[:, ::-1,:], "RGB")
            img_B = Image.fromarray(np.array(img_x)[:, ::-1,:], "RGB") """

        #self.ab = self.transform(self.ab)
        self.C = self.transform(img_c)
        self.G = self.transform(img_g)

        # return {"L": self.l, "ab": self.ab}
        return {"C": self.C,"G":self.G}

    def __len__(self):
        return len(self.imgs)

class GrayScaleDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)

        self.l = np.load(os.path.join(root,"l/gray_scale.npy"))
    def __getitem__(self, index):

        img_gray = self.l[index % len(self.l)]
        img_x = np.dstack((img_gray, img_gray, img_gray))
        img_x = Image.fromarray(img_x,'RGB')
        img_x.save("images/%s/%s.png" % ("facades", "test"))
        w, h = img_x.size
        img_A = img_x.crop((0, 0, w / 2, h))
        img_B = img_x.crop((w / 2, 0, w, h))

        if np.random.random() < 0.5: 
            img_A = Image.fromarray(np.array(img_x)[:, ::-1,:], "RGB")
            img_B = Image.fromarray(np.array(img_x)[:, ::-1,:], "RGB")

        #self.ab = self.transform(self.ab)
        self.A = self.transform(img_A)
        self.B = self.transform(img_B)

        # return {"L": self.l, "ab": self.ab}
        return {"A": self.A,"B":self.B}

    def __len__(self):
        return len(self.l)
