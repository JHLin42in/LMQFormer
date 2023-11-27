import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt


def torchPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img, 0, 1) - torch.clamp(tar_img, 0, 1)
    rmse = (imdff ** 2).mean().sqrt()
    ps = 20 * torch.log10(1 / rmse)
    return ps


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def numpyPSNR(tar_img, prd_img):
    imdff = np.float32(prd_img) - np.float32(tar_img)
    rmse = np.sqrt(np.mean(imdff ** 2))
    ps = 20 * np.log10(255 / rmse)
    return ps


def train_imshow(img, output, target):
    img = img / 2 + 0.5
    output = output / 2 + 0.5
    target = target / 2 + 0.5

    img = np.transpose(img.detach().cpu().numpy(), (1, 2, 0))
    output = np.transpose(output.detach().cpu().numpy(), (1, 2, 0))
    target = np.transpose(target.detach().cpu().numpy(), (1, 2, 0))
    plt.subplot(1, 3, 1)
    plt.title('Snow')
    plt.imshow(img)

    plt.subplot(1, 3, 2)
    plt.title('Output')
    plt.imshow(output)

    plt.subplot(1, 3, 3)
    plt.title('GT')
    plt.imshow(target)

    plt.show()


def test_imshow(img, output, target):
    img = img / 2 + 0.5
    output = output / 2 + 0.5
    target = target / 2 + 0.5

    img = np.transpose(img.detach().cpu().numpy(), (1, 2, 0))

    plt.subplot(1, 3, 1)
    plt.title('Snow')
    plt.imshow(img)

    plt.subplot(1, 3, 2)
    plt.title('Output')
    plt.imshow(output)

    plt.subplot(1, 3, 3)
    plt.title('GT')
    plt.imshow(target)

    plt.show()