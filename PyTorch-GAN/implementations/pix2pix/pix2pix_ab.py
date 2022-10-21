import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
import glob

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

# from models_depr import *
from datasets import *
from models_improved import *

import torch.nn as nn
import torch.nn.functional as F
import torch
import cv2
from skimage.color import lab2rgb

from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="improved_4", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=85, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument(
    "--sample_interval", type=int, default=200, help="interval between sampling of images from generators"
)
parser.add_argument("--checkpoint_interval", type=int, default=500, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False
print("torch.cuda.is_available()", cuda)

# Loss functions
#criterion_GAN = torch.nn.MSELoss() #model.py
criterion_GAN = torch.nn.BCEWithLogitsLoss() #test improvement
criterion_pixelwise = torch.nn.L1Loss()

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 100

# Calculate output of image discriminator (PatchGAN)
#patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)
patch = (1,30,30)

# Initialize generator and discriminator
#generator = GeneratorUNet()
#discriminator = Discriminator()
generator_a = Unet()
generator_b = Unet()

discriminator_a = PatchDiscriminator(1)
discriminator_b = PatchDiscriminator(1)

if cuda:
    generator_a = generator_a.cuda()
    generator_b = generator_b.cuda()
    discriminator_a = discriminator_a.cuda()
    discriminator_b = discriminator_b.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()

""" if opt.epoch != 0:
    # Load pretrained models
    generator_a.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch)))
    generator_b.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch)))
    discriminator.load_state_dict(torch.load("saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, opt.epoch)))
    discriminator.load_state_dict(torch.load("saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, opt.epoch)))
else: """
    # Initialize weights
generator_a.apply(weights_init_normal)
generator_b.apply(weights_init_normal)
discriminator_a.apply(weights_init_normal)
discriminator_b.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(generator_a.parameters(), generator_b.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D_a = torch.optim.Adam(discriminator_a.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_b = torch.optim.Adam(discriminator_a.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Configure dataloaders
transforms_ = [
    transforms.ToTensor(),
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.RandomHorizontalFlip()
    #transforms.Normalize((0.5), (0.5)),
]
val_transforms_ = [
    transforms.ToTensor(),
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC)
    #transforms.Normalize((0.5), (0.5)),
]

paths = glob.glob("PyTorch-GAN/data/COCO/test2017/test2017" + "/*.jpg") # Grabbing all the image file namesnp.random.seed(123)
print("Images in COCO:",len(paths))
paths_subset = np.random.choice(paths, 10000, replace=False) # choosing 10000 images randomly
rand_idxs = np.random.permutation(10000)
train_idxs = rand_idxs[:5000] # choosing the first 8000 as training set
val_idxs = rand_idxs[5000:6500] # choosing last 2000 as validation set
train_paths = paths_subset[train_idxs]
val_paths = paths_subset[val_idxs]
print("train:",len(train_paths),"val:",len(val_paths))

dataloader = DataLoader(
    CocoDataset(train_paths, transforms_=transforms_),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=0,
)

val_dataloader = DataLoader(
    CocoDataset(val_paths, transforms_=val_transforms_, mode="val"),
    batch_size=10,
    shuffle=True,
    num_workers=0,
)

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def lab_to_rgb(L, a,b):
    """
    Takes a batch of images
    """
    #ab = torch.cat((a,b),1)
    #print("\nab:",ab.shape)
    L = (L + 1.) * 50.
    a = a * 110.
    b = b * 110.
    #print("\nL:",L.shape, "\n")
    Lab = torch.cat([L, a, b], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    #print("\nlab:",Lab.shape, "\n")
    rgb_imgs = []
    for img in Lab:
        #img_rgb=cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)

def sample_images(batches_done,save=True):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    input = Variable(imgs["L"].type(Tensor))
    real_a = Variable(imgs["a"].type(Tensor))
    real_b = Variable(imgs["b"].type(Tensor))
    
    fake_C_a = generator_a(input).detach()
    fake_C_b = generator_b(input).detach()

    real_imgs = lab_to_rgb(input,real_a,real_b)
    fake_imgs = lab_to_rgb(input,fake_C_a,fake_C_b)
    fig = plt.figure(figsize=(30, 16))
    for i in range(5):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(input[i][0].cpu(), cmap='gray')
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(fake_imgs[i])
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 10)
        ax.imshow(real_imgs[i])
        ax.axis("off")
    if save:
        fig.savefig(f"images/{opt.dataset_name}/colorization_{batches_done}.png")
    
    


# ----------
#  Training
# ----------

prev_time = time.time()
replay_buffer = [] #Buffer for the replays

for epoch in range(opt.epoch, opt.n_epochs):
    print(epoch, "epoch")
    torch.cuda.empty_cache()
    for i, batch in enumerate(dataloader):
        # Model inputs
        real_a = Variable(batch["a"].type(Tensor))
        real_b = Variable(batch["b"].type(Tensor))
        # real_a = real_C[[0,1],...]
        # real_b = real_C[[0,2],...]
        input =Variable(batch["L"].type(Tensor))
        #print(("\nreal a:",real_a.shape), ('real b: ',real_b.shape), "\n")

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_a.size(0), *patch))*0.9), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_a.size(0), *patch))), requires_grad=False)
        """ valid = Variable(Tensor(real_C.shape[0], 1).fill_(0.9), requires_grad=False)
        fake = Variable(Tensor(real_C.shape[0], 1).fill_(0.0), requires_grad=False)
 """
        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # GAN loss
        #generate noise
        #z = Variable(Tensor(np.random.normal(0, 1, (input.shape[0], 100))))

        fake_a = generator_a(input)
        fake_b = generator_b(input)
        #fake_image = torch.cat([input,fake_C],dim=1)
        pred_fake_a = discriminator_a(fake_a)
        pred_fake_b = discriminator_b(fake_b)
        

        loss_GAN= criterion_GAN(pred_fake_a, valid) + criterion_GAN(pred_fake_b, valid)
        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(fake_a, real_a) + criterion_pixelwise(fake_b, real_b)
        # Total loss
        loss_G = loss_GAN + lambda_pixel * loss_pixel

        loss_G.backward()

        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D_a.zero_grad()
        optimizer_D_b.zero_grad()

        # Real loss
        #real_image = torch.cat([input,real_C],dim=1)
        pred_real_a = discriminator_a(real_a)
        pred_real_b = discriminator_b(real_b)
        loss_real_a = criterion_GAN(pred_real_a, valid)
        loss_real_b = criterion_GAN(pred_real_b, valid)


####Experience replay with 20% chance##########
        # Fake loss
        # if np.random.uniform() < 0.2 and len(replay_buffer) > 0:
        #     old_index = np.random.choice(len(replay_buffer))
        #     fake_old = replay_buffer.pop(old_index)
        #     pred_old = discriminator(fake_old.cuda())
        #     fake = Variable(Tensor(np.zeros((fake_old.size(0), *patch))), requires_grad=False)
        #     loss_old = criterion_GAN(pred_old, fake)
            
        #     loss_D =  (loss_real + loss_old) / 2
        # else:
        #     fake_image = torch.cat([input,fake_C],dim=1)
        #     pred_fake = discriminator(fake_image.detach())
        #     loss_fake = criterion_GAN(pred_fake, fake)
        #     if np.random.uniform() < 0.4:
        #         if len(replay_buffer) < 300:
        #             replay_buffer.append(fake_image.detach().cpu())
        #     loss_D =  (loss_real + loss_fake) / 2
        #fake_image = torch.cat([input,fake_C],dim=1)
        pred_fake_a = discriminator_a(fake_a).detach()
        pred_fake_b = discriminator_b(fake_b).detach()
        loss_fake_a = criterion_GAN(pred_fake_a, fake)
        loss_fake_b = criterion_GAN(pred_fake_b, fake)
        loss_D_a =  (loss_real_a + loss_fake_a) / 2
        loss_D_b =  (loss_real_b + loss_fake_b) / 2
        # Total loss
        loss_D = loss_D_a + loss_D_b

        loss_D.backward()
        optimizer_D_a.step()
        optimizer_D_b.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss a: %f] [D loss b: %f] [G loss: %f, pixel: %f, adv_a: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D_a.item(),
                loss_D_b.item(),
                loss_G.item(),
                loss_pixel.item(),
                loss_GAN.item(),
                time_left,
            )
        )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

    #if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
torch.save(generator_a.state_dict(), "saved_models/%s/generator_a_%d.pth" % (opt.dataset_name, epoch))
torch.save(generator_b.state_dict(), "saved_models/%s/generator_b_%d.pth" % (opt.dataset_name, epoch))
torch.save(discriminator_a.state_dict(), "saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, epoch))
