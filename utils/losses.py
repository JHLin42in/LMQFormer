import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from math import log


# ----------------------------------------------------------------------------------------------------------------------
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)  # filter
        down = filtered[:, :, ::2, ::2]  # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4  # upsample
        filtered = self.conv_gauss(new_filter)  # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        '''
         use vgg19 conv1_2, conv2_2, conv3_3 feature, before relu layer
        '''
        self.feature_list = [2, 7, 14]
        vgg19 = torchvision.models.vgg19(pretrained=True).cuda()
        self.model = nn.Sequential(*list(vgg19.features.children())[:self.feature_list[-1] + 1])

    def forward(self, x):
        x = (x - 0.5) / 0.5
        features = []
        for i, layer in enumerate(list(self.model)):
            x = layer(x)
            if i in self.feature_list:
                features.append(x)

        return features


class CCPLoss(nn.Module):
    def __init__(self):
        super(CCPLoss, self).__init__()
        self.pool1 = nn.MaxPool2d(35, 1, 35//2)
        self.pool2 = nn.MaxPool2d(35, 1, 35//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, restored, target):
        restored, _ = torch.min(restored, dim=1)
        restored = self.pool1(restored)

        target, _ = torch.min(target, dim=1)
        target = self.pool2(target)
        out = torch.mean(torch.abs(restored-target))
        out = self.sigmoid(out)
        return out


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg19 = VGG19()

    def forward(self, x, y):
        weights = [1, 0.2, 0.04]
        # 得到fake与real的特征
        features_fake = self.vgg19(x)
        features_real = self.vgg19(y)
        features_real_no_grad = [f_real.detach() for f_real in features_real]
        mse_loss = nn.MSELoss(reduction='elementwise_mean')
        # 计算两者的loss
        loss = 0
        for i in range(len(features_real)):
            loss_i = mse_loss(features_fake[i], features_real_no_grad[i])
            loss = loss + loss_i * weights[i]
        return loss


# ----------------------------------------------------------------------------------------------------------------------
class laplace_filter(nn.Module):
    def __init__(self):
        super().__init__()
        self.laplace = torch.tensor([[0, -1, 0],
                                    [-1, 4, -1],
                                    [0, -1, 0]],
                                    dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)
        if torch.cuda.is_available():
            self.laplace = self.laplace.cuda()
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.conv.weight.data = self.laplace
        self.relu = nn.ReLU(inplace=True)

    def forward(self, snow, clean):
        snow = torch.mean(snow, dim=1, keepdim=True)
        clean = torch.mean(clean, dim=1, keepdim=True)
        clean_lap = self.relu(self.conv(clean)).detach()
        snow_lap = self.relu(self.conv(snow)).detach()
        mask_lap = snow_lap - clean_lap
        mask_lap = self.relu(mask_lap)
        return snow_lap, clean_lap, mask_lap


class criterion_former(nn.Module):
    def __init__(self, lambda1=0.05, lambda2=0.1):
        super().__init__()
        self.criterion_char = CharbonnierLoss()
        self.criterion_edge = EdgeLoss()
        self.criterion_per = PerceptualLoss()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def forward(self, restored, target):
        loss_char = self.criterion_char(restored, target)
        loss_edge = self.criterion_edge(restored, target)
        loss_per = self.criterion_per(restored, target)
        return loss_char + self.lambda1 * loss_edge + self.lambda2 * loss_per


class criterion_vqvae(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.criterion_edge = EdgeLoss()

    def forward(self, vaemask, mask_lap, vaeclean, clean_lap, latent_loss):
        loss_mse1 = self.mse(vaemask, mask_lap)
        loss_mse2 = self.mse(vaeclean, clean_lap)

        return 2*(loss_mse1 + loss_mse2), latent_loss.mean()
