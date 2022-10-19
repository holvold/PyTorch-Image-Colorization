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
from torchvision.utils import save_image
from skimage.color import lab2rgb,rgb2lab
from matplotlib import pyplot as plt

class ImageDataset(Dataset):

    def clear_black_white(self,images):
        saved_images = []
        saved_bw = []
        for i,img in enumerate(images):
            l, a, b = cv2.split(img)
            diff = np.mean(np.abs(a - b))
            if diff > 40: 
                saved_images.append(img)
                saved_bw.append(np.dstack((l, np.zeros(l.shape), np.zeros(l.shape))))
        return saved_images,saved_bw

    def __init__(self, root, transforms_=None, mode='train',):
        self.transform = transforms.Compose(transforms_)

        self.l = np.load(os.path.join(root,"l/gray_scale.npy"))
        ab1 = np.load(os.path.join(root,"ab", "ab1.npy"))
        ab2 = np.load(os.path.join(root, "ab", "ab2.npy"))
        ab3 = np.load(os.path.join(root,"ab", "ab3.npy"))
        self.ab = np.concatenate([ab1, ab2, ab3], axis=0)
        if mode=="val":
            self.imgs = np.concatenate((np.expand_dims(self.l,axis=-1),self.ab), axis=3)[4001:5000]
        else:
            self.imgs = np.concatenate((np.expand_dims(self.l,axis=-1),self.ab), axis=3)[:4000]
        self.imgs,self.l = self.clear_black_white(self.imgs)
        print(len(self.imgs),len(self.l))

        self.l = np.load(os.path.join(root,"l/gray_scale.npy"))
    def __getitem__(self, index):

        img_x = self.imgs[index % len(self.imgs)]
        img_x[...,0] = (img_x[...,0] +1) *50
        #img_bw = self.imgs[index % len(self.imgs)]
        #img_bw[...,1] = img_bw[...,2] = 0
        img_bw = lab2rgb(img_x)
        plt.figure(figsize=(20,10))
        plt.subplot(121), plt.imshow(lab2rgb(img_x)), plt.axis('off'), plt.title('Original image', size=20)
        plt.subplot(122), plt.imshow(img_bw), plt.axis('off'), plt.title('Gray scale image', size=20)
        plt.show()
        #img_rgb = cv2.cvtColor(img_x,cv2.COLOR_Lab2RGB)
        #img_c = Image.fromarray(img_x,'LAB')
        img_g = self.l[index % len(self.l)]
        #img_g = Image.fromarray(img_g,'LAB')
        #w, h = img_x.size
        """ img_A = img_x.crop((0, 0, w / 2, h))
        img_B = img_x.crop((w / 2, 0, w, h)) """

        """ if np.random.random() < 0.5: 
            img_c = Image.fromarray(np.array(img_c)[:, ::-1,:], "LAB")
            img_g = Image.fromarray(np.array(img_g)[:, ::-1,:], "LAB") """

        #self.ab = self.transform(self.ab)
        self.C = self.transform(img_x)
        self.G = self.transform(img_g)

        # return {"L": self.l, "ab": self.ab}
        return {"C": self.C,"G":self.G}

    def __len__(self):
        return len(self.imgs)

class CocoDataset(Dataset):
    def black_and_white(self,image):
        saved_images = []
        l, a, b = cv2.split(np.array(image))
        diff = np.mean(np.abs(a - b))
        if diff < 40: 
            return True
            saved_images.append(img)
        return False

    def __init__(self, paths, transforms_=None, mode="train"):
        self.transforms = transforms.Compose(transforms_)

        self.paths = [path for path in paths if not self.black_and_white(cv2.imread(path))]

        print(len(self.paths))

    def __getitem__(self, index):

        img = Image.open(self.paths[index]).convert("RGB")
        img = self.transforms(img)
        # img.save(f"images/test/colorization_test.png")
        #save_image(img, "images/%s/%s.png" % ('test','test'), nrow=5, normalize=True)

        img = np.array(torch.permute(img,(1,2,0)))
        #print("img.shape:",img.shape)
        #plt.imshow(img) #Needs to be in row,col order
        #plt.savefig("images/test/colorization_test2.png")
        #img = Image.fromarray(img)
        #img = np.array(img)
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB) # Converting RGB to L*a*b
        """ print(np.max(img_rgb),np.min(img_rgb))
        plt.imshow(img_rgb) #Needs to be in row,col order
        plt.savefig("images/test/colorization_test3.png") """
        #img_lab = rgb2lab(img)
        #print('lab shape', img_lab.shape)
        #img_rgb = lab2rgb(img_lab)
        
        img_lab = transforms.ToTensor()(img_lab)
        #print("tensor.img_lab.shape:",img_lab.shape)
        #save_image(img_rgb, "images/%s/%s.png" % ('test','testlab'))
        L = img_lab[[0], ...] / 50. - 1. # Between -1 and 1
        ab = img_lab[[1, 2], ...] / 110. # Between -1 and 1
        
        return {'L': L, 'ab': ab}

    def __len__(self):
        return len(self.paths)
