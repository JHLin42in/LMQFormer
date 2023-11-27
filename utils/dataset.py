import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import random


def get_training_data(rgb_dir, img_options, aug):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, img_options, aug)


def get_validation_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, img_options)


def get_test_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, img_options)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif', 'tif'])


class MixUp_AUG:
    def __init__(self):
        self.dist = torch.distributions.beta.Beta(torch.tensor([0.6]), torch.tensor([0.6]))

    def aug(self, rgb_gt, rgb_noisy):
        bs = rgb_gt.size(0)
        indices = torch.randperm(bs)
        rgb_gt2 = rgb_gt[indices]
        rgb_noisy2 = rgb_noisy[indices]

        lam = self.dist.rsample((bs,1)).view(-1,1,1,1).cuda()

        rgb_gt    = lam * rgb_gt + (1-lam) * rgb_gt2
        rgb_noisy = lam * rgb_noisy + (1-lam) * rgb_noisy2

        return rgb_gt, rgb_noisy


class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None, aug=False, transform=False):
        super(DataLoaderTrain, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'Snow')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'Gt')))

        self.inp_filenames = [os.path.join(rgb_dir, 'Snow', x)  for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'Gt', x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex       = len(self.tar_filenames)  # get the size of Gt

        self.ps = self.img_options['patch_size']
        self.aug = aug
        self.transform = transform

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_path1 = open(inp_path, 'rb')
        tar_path1 = open(tar_path, 'rb')
        inp_img = Image.open(inp_path1)
        tar_img = Image.open(tar_path1)

        w,h = tar_img.size
        padw = ps-w if w<ps else 0
        padh = ps-h if h<ps else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw!=0 or padh!=0:
            inp_img = TF.pad(inp_img, (0,0,padw,padh), padding_mode='reflect')
            tar_img = TF.pad(tar_img, (0,0,padw,padh), padding_mode='reflect')

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        hh, ww = tar_img.shape[1], tar_img.shape[2]

        rr     = random.randint(0, hh-ps)
        cc     = random.randint(0, ww-ps)
        aug    = random.randint(0, 8)

        # Crop patch
        inp_img = inp_img[:, rr:rr+ps, cc:cc+ps]
        tar_img = tar_img[:, rr:rr+ps, cc:cc+ps]

        # Data Augmentations
        if self.aug == True:
            if aug==1:
                inp_img = inp_img.flip(1)
                tar_img = tar_img.flip(1)
            elif aug==2:
                inp_img = inp_img.flip(2)
                tar_img = tar_img.flip(2)
            elif aug==3:
                inp_img = torch.rot90(inp_img,dims=(1,2))
                tar_img = torch.rot90(tar_img,dims=(1,2))
            elif aug==4:
                inp_img = torch.rot90(inp_img,dims=(1,2), k=2)
                tar_img = torch.rot90(tar_img,dims=(1,2), k=2)
            elif aug==5:
                inp_img = torch.rot90(inp_img,dims=(1,2), k=3)
                tar_img = torch.rot90(tar_img,dims=(1,2), k=3)
            elif aug==6:
                inp_img = torch.rot90(inp_img.flip(1),dims=(1,2))
                tar_img = torch.rot90(tar_img.flip(1),dims=(1,2))
            elif aug==7:
                inp_img = torch.rot90(inp_img.flip(2),dims=(1,2))
                tar_img = torch.rot90(tar_img.flip(2),dims=(1,2))
        
        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        inp_path1.close()
        tar_path1.close()
        return tar_img, inp_img, filename


class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, img_options=None):
        super(DataLoaderVal, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'Snow')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'Gt')))

        self.inp_filenames = [os.path.join(rgb_dir, 'Snow', x)  for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'Gt', x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex       = len(self.tar_filenames)  # get the size of Gt

        if img_options is not None:
            self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_path1 = open(inp_path, 'rb')
        tar_path1 = open(tar_path, 'rb')
        inp_img = Image.open(inp_path1)
        tar_img = Image.open(tar_path1)

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        H = inp_img.shape[-2]
        W = inp_img.shape[-1]

        # Validate on center crop
        if self.img_options is not None:
            ps = self.ps
            inp_img = TF.center_crop(inp_img, (ps,ps))
            tar_img = TF.center_crop(tar_img, (ps,ps))

        elif self.img_options is None:
            if H % 32 != 0:
                new_H = H // 32 * 32
            else:
                new_H = H
            if W % 32 != 0:
                new_W = W // 32 * 32
            else:
                new_W = W
            inp_img = TF.resize(inp_img, (new_H, new_W))
            # tar_img = TF.resize(tar_img, (new_H, new_W))

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]
        inp_path1.close()
        tar_path1.close()

        return tar_img, inp_img, filename, H, W


class DataLoaderTest(Dataset):
    def __init__(self, inp_dir, img_options):
        super(DataLoaderTest, self).__init__()

        inp_files = sorted(os.listdir(inp_dir))
        self.inp_filenames = [os.path.join(inp_dir, x) for x in inp_files if is_image_file(x)]

        self.inp_size = len(self.inp_filenames)
        self.img_options = img_options

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):
        path_inp = self.inp_filenames[index]
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]
        fileformat = os.path.splitext(os.path.split(path_inp)[-1])[-1]
        inp = Image.open(path_inp)
        inp = TF.to_tensor(inp)
        H = inp.shape[-2]
        W = inp.shape[-1]
        if H % 32 != 0:
            new_H = H // 32 * 32
        else:
            new_H = H
        if W % 32 != 0:
            new_W = W // 32 * 32
        else:
            new_W = W

        inp = TF.resize(inp, (new_H, new_W))

        return inp, filename, fileformat, H, W
