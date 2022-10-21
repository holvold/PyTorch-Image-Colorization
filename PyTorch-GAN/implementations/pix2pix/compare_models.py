import models_improved as imp
import models_baseline as base
from datasets import *
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from skimage.color import lab2rgb

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

base_gen = base.Unet()

imp_gen_a = imp.Unet()
imp_gen_b = imp.Unet()

if cuda:
    base_gen = base_gen.cuda()
    imp_gen_a = imp_gen_a.cuda()
    imp_gen_b = imp_gen_b.cuda()

base_gen.load_state_dict(torch.load("saved_models/facades/generator_99.pth"))

imp_gen_a.load_state_dict(torch.load("saved_models/improved/generator_99.pth"))
imp_gen_b.load_state_dict(torch.load("saved_models/improved/generator_99.pth"))

paths = glob.glob("PyTorch-GAN/data/COCO/test2017/test2017" + "/*.jpg") # Grabbing all the image file namesnp.random.seed(123)
paths_subset = np.random.choice(paths, 10000, replace=False) # choosing 10000 images randomly
rand_idxs = np.random.permutation(10000)
train_idxs = rand_idxs[:4000] # choosing the first 8000 as training set
val_idxs = rand_idxs[4000:5000] # choosing last 2000 as validation set
train_paths = paths_subset[train_idxs]
val_paths = paths_subset[val_idxs]


val_transforms_ = [
    transforms.ToTensor(),
    transforms.Resize((256, 256), Image.BICUBIC),
]

val_dataloader = DataLoader(
    CocoDataset(val_paths, transforms_=val_transforms_, mode="val"),
    batch_size=10,
    shuffle=True,
    num_workers=0,
)

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
    real_C = Variable(imgs["ab"].type(Tensor))
    fake_base = base_gen(input).detach()
    fake_impr_a = imp_gen_a(input).detach()
    fake_impr_b = imp_gen_b(input).detach()
    
    real_imgs = lab_to_rgb(input,real_C)
    fake_base = lab_to_rgb(input,fake_base)
    fake_impr = lab_to_rgb(input,fake_impr_a, fake_impr_b)
    fig = plt.figure(figsize=(30, 16))
    for i in range(5):
        ax = plt.subplot(4, 5, i + 1 + 1)
        ax.imshow(fake_base[i])
        ax.axis("off")
        ax = plt.subplot(4, 5, i + 1 + 5)
        ax.imshow(fake_impr[i])
        ax.axis("off")
        ax = plt.subplot(4, 5, i + 1 + 10)
        ax.imshow(real_imgs[i])
        ax.axis("off")
    if save:
        fig.savefig(f"images/compare/base_improved4_{batches_done}.png")

for i in range(5):
    sample_images(i)