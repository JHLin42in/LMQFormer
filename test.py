# -*- coding: utf-8 -*-
# @Time    : 2022/3/4 9:14
# @Author  : Lin Junhong
# @FileName: test.py
# @Software: PyCharm
# @E_mails ：SPJLinn@163.com

import argparse
import os
import time

import PIL.Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from utils.dataset import get_test_data, get_validation_data
from models.LMQFormer import LMQFORMER as Net
# from checkpoints.Ablation.VAE.VAE import LMQFORMER as Net


# ======================================================================================================================
weights_dir = './checkpoints/Snow100K.pth'

data_dir = './testset/'
result_dir = './results/LMQFormer/real1000/'

if os.path.isfile(data_dir):
    Scenario = 'single'
elif os.path.isfile(data_dir + os.listdir(data_dir)[0]):
    Scenario = 'real'
else:
    Scenario = 'synth'

data_dir_str = data_dir.split('/')[2:-1]
data_dir_str = '/'.join(data_dir_str)

parser = argparse.ArgumentParser(description='Single Image Snow Removal')
parser.add_argument('--input_dir', default=data_dir, type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default=result_dir, type=str, help='Directory for results')
parser.add_argument('--weights_dir', default=weights_dir, type=str, help='Path to weights')
parser.add_argument('--scenario', default=Scenario, type=str, help='Different test Scenario')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--device', default='cuda', type=str, help='cuda or cpu')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


# ======================================================================================================================
class TEST:
    def __init__(self, weights_dir):
        self.model = Net().to(args.device)
        utils.load_checkpoint(self.model, weights_dir)
        self.model.eval()
        # self.model = nn.DataParallel(self.model)
        # macs, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True,
        #                                          print_per_layer_stat=False, verbose=True)

        result_dir = os.path.join(args.result_dir)
        utils.mkdir(result_dir)

        print('====> Scenario: ', Scenario)
        # print(f'====> Params: {params}  Macs: {macs}')
        print(f'====> Testing weights: {weights_dir}')
        print('====> Save Dir: ', result_dir)

    def single_img(self, data_dir, result_dir, save_featmaps=False, imgname='VAE'):
        img_name = data_dir.split('/')[-1].split('.')[0]
        img = Image.open(data_dir)
        img = TF.to_tensor(img)
        H = img.shape[-2]
        W = img.shape[-1]
        if H % 32 != 0:
            new_H = H // 32 * 32
        else:
            new_H = H

        if W % 32 != 0:
            new_W = W // 32 * 32
        else:
            new_W = W

        COLORMODE = cv2.COLORMAP_TWILIGHT_SHIFTED
        # img = TF.resize(img, (new_H, new_W))
        img = TF.center_crop(img, (256, 256))
        utils.save_img((os.path.join(result_dir, img_name + '_origin.png')), img_as_ubyte(img))
        with torch.no_grad():
            # torch.cuda.ipc_collect()
            # torch.cuda.empty_cache()
            input_ = torch.unsqueeze(img, 0).to(args.device)
            # img = torch.clamp(img, 0, 1).permute(1, 2, 0).cpu().detach().numpy()
            # utils.save_img((os.path.join(result_dir, img_name + '.png')), img_as_ubyte(img))

            start_time = time.time()
            clean, res = self.model(input_)
            end_time = time.time()
            pred_time = end_time - start_time

            # res = TF.resize(res[0], (new_H, new_W))
            # res = torch.clamp(res, 0, 1).permute(1, 2, 0).cpu().detach().numpy()

            # utils.save_img((os.path.join(result_dir, imgname + '_res.png')), img_as_ubyte(res))
            utils.save_img((os.path.join(result_dir, imgname + '_clean.png')), img_as_ubyte(clean))

        print(f"One Img Time: {pred_time}")

    def norefer(self, data_dir, result_dir):
        test_dataset = get_test_data(data_dir, img_options={})
        test_loader = DataLoader(dataset=test_dataset, batch_size=1,
                                 shuffle=False, num_workers=0, drop_last=False, pin_memory=True)
        print('Img Number: ', len(test_loader))

        with torch.no_grad():
            time_count = 0

            pbar = tqdm(test_loader)
            for ii, data_test in enumerate(pbar, 0):
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()

                input_ = data_test[0].to(args.device)
                filenames = data_test[1]
                fileformat = data_test[2]
                H, W = int(data_test[3][0]), int(data_test[4][0])

                start_time = time.time()
                clean, res = self.model(input_)
                end_time = time.time()
                pred_time = end_time - start_time

                clean = TF.resize(clean, (H, W), interpolation=PIL.Image.BICUBIC)
                clean = torch.clamp(clean, 0, 1).permute(0, 2, 3, 1).cpu().detach().numpy()
                # res = TF.resize(res, (H, W), interpolation=PIL.Image.BICUBIC)
                # res = torch.clamp(res, 0, 1).permute(0, 2, 3, 1).cpu().detach().numpy()

                time_count += pred_time

                for batch in range(len(clean)):
                    utils.save_img((os.path.join(result_dir, f'{filenames[batch]}_clean{fileformat[batch]}')), img_as_ubyte(clean[batch]))
                    # utils.save_img((os.path.join(result_dir, f'{filenames[batch]}_res{fileformat[batch]}')), img_as_ubyte(res[batch]))

                info = f'One Img Time:{pred_time:.4f} Total Processing Time:{time_count:.4f} '
                pbar.set_description(info)

        img_per_sec = time_count / (len(test_loader))
        print(f"Avg One Img Time: {img_per_sec}")

    def refer(self, data_dir, result_dir):
        # img_options={'patch_size': 256}
        val_dataset = get_validation_data(data_dir, img_options=None)
        val_loader = DataLoader(dataset=val_dataset, batch_size=1,
                                shuffle=False, num_workers=0, drop_last=False, pin_memory=True)
        print('Img Number: ', len(val_loader) * 1)

        with torch.no_grad():
            psnr_val_rgb = []
            ssim_val_rgb = []
            time_count = 0
            PSNRS = 0
            SSIMS = 0
            pbar = tqdm(val_loader)
            for ii, data_val in enumerate(pbar, 0):

                target = data_val[0].to(args.device)
                input_ = data_val[1].to(args.device)
                filenames = data_val[2]
                H, W = int(data_val[3][0]), int(data_val[4][0])

                start_time = time.time()
                clean, res = self.model(input_)
                end_time = time.time()
                pred_time = end_time - start_time
                time_count = time_count + pred_time

                clean = TF.resize(clean, (H, W))
                # res = TF.resize(res, (H, W))

                # clean = torch.clamp(clean, 0, 1).permute(0, 2, 3, 1).cpu().detach().numpy()

                clean = torch.clamp(clean, 0, 1).cpu().numpy().squeeze().transpose(1, 2, 0)
                target = torch.clamp(target, 0, 1).cpu().numpy().squeeze().transpose(1, 2, 0)
                # res = torch.clamp(res, 0, 1).cpu().numpy().squeeze().transpose(1, 2, 0)

                # for batch in range(len(clean)):
                psnr = psnr_loss(clean, target)
                ssim = ssim_loss(clean, target, multichannel=True)
                PSNRS += psnr
                SSIMS += ssim
                psnr_val_rgb.append(psnr)
                ssim_val_rgb.append(ssim)
                utils.save_img((os.path.join(result_dir, filenames[0] + '_clean.png')), img_as_ubyte(clean))
                # utils.save_img((os.path.join(result_dir, filenames[0] + '_res.png')), img_as_ubyte(res))

                info = f'One Batch Time:{pred_time:.4f} Image:{H, W} PSNR:{psnr:.4f} SSIM:{ssim:.4f} ' \
                       f'Total Processing Time:{time_count:.4f} '
                pbar.set_description(info)
            # print(f'Avg PSNR: {PSNRS/len(val_loader)} SSIM: {SSIMS/len(val_loader)}')
        img_per_sec = time_count / (len(val_loader) * 1)
        psnr_val_rgb = sum(psnr_val_rgb) / (len(val_loader) * 1)
        ssim_val_rgb = sum(ssim_val_rgb) / (len(val_loader) * 1)
        print(f"Avg One Img Time: {img_per_sec} \nPSNR: {psnr_val_rgb}  SSIM: {ssim_val_rgb}")


# ======================================================================================================================
if __name__ == '__main__':
    Test = TEST(weights_dir=weights_dir)
    Test.norefer(data_dir, result_dir)
