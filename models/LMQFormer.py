# -*- coding: utf-8 -*-
# @Time    : 2022/8/3 19:14
# @Author  : Lin Junhong
# @FileName: LMQFormer.py
# @Software: PyCharm
# @E_mails ：SPJLinn@163.com


import torch
from torch import nn
from torch.nn import functional as F
from timm.models.layers import to_2tuple, DropPath
from ptflops import get_model_complexity_info
import utils.distributed as dist_fn


# ======================================================================================================================
class units():
    class Laplace(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.laplace = torch.tensor([[0, -1, 0],
                                         [-1, 4, -1],
                                         [0, -1, 0]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)

            self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.conv.weight.data = self.laplace.repeat(1, channels, 1, 1)
            self.conv.requires_grads = False
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.conv(x).detach()

    class ChannelAttention(nn.Module):
        def __init__(self, k_size=7):
            super().__init__()
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.conv1d = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size // 2), bias=False)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            y = self.avgpool(x)
            y = self.conv1d(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            y = self.sigmoid(y.float())
            return x * y.expand_as(x)

    class SpatialAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            avgout = torch.mean(x, dim=1, keepdim=True)
            maxout, _ = torch.max(x, dim=1, keepdim=True)
            y = torch.cat([avgout, maxout], dim=1)
            y = self.conv(y)
            return x * self.sigmoid(y)

    @staticmethod
    def PDConv(channels, k_size, groups):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=k_size, padding=(k_size // 2), bias=False, groups=groups),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        )

    @staticmethod
    def NormAct(channels):
        return nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(inplace=True)
        )

    @staticmethod
    def Squeeze(channels):
        return nn.Conv2d(channels * 2, channels, 1, bias=False)

    @staticmethod
    def ch_shuffle(x, groups):
        B, C, H, W = x.data.size()
        x = x.view(B, groups, C // groups, H, W)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(B, -1, H, W)
        return x

    @staticmethod
    def UpDownSample(ch_in, ch_out, scale):
        return nn.Sequential(nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False),
                             nn.Conv2d(ch_in, ch_out, kernel_size=1, bias=False))


units = units()


# ----------------------------------------------------------------------------------------------------------------------
class GLP_VQVAE(nn.Module):
    def __init__(self, ch_in=1, channels=24, embed_dim=24, n_embed=256, decay=0.1, groups=1):
        super().__init__()
        self.Laplace_op = self.Laplace()
        self.img_in = nn.Conv2d(ch_in, channels, kernel_size=1, bias=False)
        self.img_out = nn.Conv2d(channels, ch_in, kernel_size=1, bias=False)

        self.encoder = self.EnDecoder(channels=channels, groups=groups, code='en')
        self.decoder = self.EnDecoder(channels=channels, groups=groups, code='de')

        # self.Qt1 = Quantize(channels=channels, embed_dim=embed_dim, n_embed=n_embed, decay=decay)
        self.Qt2 = self.Quantize(channels=channels, embed_dim=embed_dim, n_embed=n_embed, decay=decay)
        self.Qt3 = self.Quantize(channels=channels, embed_dim=embed_dim, n_embed=n_embed, decay=decay)

    class Laplace(nn.Module):
        def __init__(self):
            super().__init__()
            self.laplace = torch.tensor([[0, -1, 0],
                                         [-1, 4, -1],
                                         [0, -1, 0]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)

            self.conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)
            self.conv.weight.data = self.laplace  # .repeat(1, 3, 1, 1)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            x = torch.mean(x, dim=1, keepdim=True)
            return self.conv(x).detach()

    class Quantize(nn.Module):
        def __init__(self, channels, embed_dim, n_embed, decay=0., eps=1e-5):
            super().__init__()

            self.dim = embed_dim
            self.n_embed = n_embed
            self.decay = decay
            self.eps = eps

            embed = torch.randn(embed_dim, n_embed)
            self.register_buffer("embed", embed)
            self.register_buffer("cluster_size", torch.zeros(n_embed))
            self.register_buffer("embed_avg", embed.clone())

        def forward(self, input):
            flatten = input.reshape(-1, self.dim)
            dist = (
                    flatten.pow(2).sum(1, keepdim=True)
                    - 2 * flatten @ self.embed
                    + self.embed.pow(2).sum(0, keepdim=True)
            )
            _, embed_ind = (-dist).max(1)
            embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
            embed_ind = embed_ind.view(*input.shape[:-1])

            quantize = self.embed_code(embed_ind)

            if self.training:
                embed_onehot_sum = embed_onehot.sum(0)
                embed_sum = flatten.transpose(0, 1) @ embed_onehot

                dist_fn.all_reduce(embed_onehot_sum)
                dist_fn.all_reduce(embed_sum)

                self.cluster_size.data.mul_(self.decay).add_(
                    embed_onehot_sum, alpha=1 - self.decay
                )
                self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
                n = self.cluster_size.sum()
                cluster_size = (
                        (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
                )
                embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
                self.embed.data.copy_(embed_normalized)

            diff = (quantize.detach() - input).pow(2).mean()

            quantize = input + (quantize - input).detach()
            return quantize, diff, embed_ind

        def embed_code(self, embed_id):
            return F.embedding(embed_id, self.embed.transpose(0, 1))

    class EnDecoder(nn.Module):
        def __init__(self, channels, groups, code):
            super().__init__()
            self.code = code
            if code == 'en':
                self.top = self.Block(channels, k_size=7, groups=groups, code=code)
                self.middle = self.Block(channels, k_size=5, groups=groups, code=code)
                self.down = self.Block(channels, k_size=3, groups=groups, code=code)

            elif code == 'de':
                self.top = self.Block(channels, k_size=7, groups=groups, code=code)
                self.middle = self.Block(channels, k_size=5, groups=groups, code=code)
                self.down = self.Block(channels, k_size=3, groups=groups, code=code)

                self.squ2 = units.Squeeze(channels)
                self.squ1 = units.Squeeze(channels)

            self.res = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False, groups=groups),
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                units.SpatialAttention(),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            if self.code == 'en':
                enc1 = self.top(x)
                enc2 = self.middle(enc1)
                enc3 = self.down(enc2)

                return enc1, enc2, enc3

            elif self.code == 'de':
                enc1, enc2, enc3 = x
                dec3 = self.down(enc3)

                dec2 = self.squ2(torch.cat((dec3, enc2), dim=1))
                dec2 = self.middle(dec2)

                dec1 = self.squ1(torch.cat((dec2, enc1), dim=1))
                dec1 = self.top(dec1)

                return dec1

        class Block(nn.Module):
            def __init__(self, channels, k_size, groups, code):
                super().__init__()
                self.code = code
                self.res = nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=k_size, padding=(k_size // 2), bias=False, groups=groups),
                    nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                    units.SpatialAttention(),
                    nn.ReLU(inplace=True)
                )

                if code == 'en':
                    self.out = nn.Conv2d(channels, channels, kernel_size=4,
                                         stride=2, padding=1, bias=False, groups=groups)
                elif code == 'de':
                    self.out = nn.ConvTranspose2d(channels, channels, kernel_size=4,
                                                  stride=2, padding=1, bias=False, groups=groups)

            def forward(self, x):
                if self.code == 'en':
                    res = self.res(x)
                    x = x + res
                    x = self.out(x)
                elif self.code == 'de':
                    x = self.out(x)
                    res = self.res(x)
                    x = x + res
                return x

    def forward(self, input):
        input_lap = self.Laplace_op(input)
        input = self.img_in(input_lap)
        enc1, enc2, enc3 = self.encoder(input)

        # qt1, diff1, id1 = self.Qt1(enc1.permute(0, 2, 3, 1))
        qt2, diff2, id2 = self.Qt2(enc2.permute(0, 2, 3, 1))
        qt3, diff3, id3 = self.Qt3(enc3.permute(0, 2, 3, 1))

        # qt1 = qt1.permute(0, 3, 1, 2)
        # diff1 = diff1.unsqueeze(0)
        qt2 = qt2.permute(0, 3, 1, 2)
        diff2 = diff2.unsqueeze(0)
        qt3 = qt3.permute(0, 3, 1, 2)
        diff3 = diff3.unsqueeze(0)

        dec = self.decoder((enc1, qt2, qt3))
        mask_lap = self.img_out(dec)
        clean_lap = input_lap - mask_lap

        return mask_lap, clean_lap, diff2 + diff3


# ----------------------------------------------------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, channels, mlp_ratio=1):
        super().__init__()

        self.pfc1 = nn.Conv2d(channels, int(channels * mlp_ratio), kernel_size=1, bias=False)
        self.act = nn.LeakyReLU(inplace=True)
        self.pfc2 = nn.Conv2d(int(channels * mlp_ratio), channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.pfc2(self.act(self.pfc1(x)))


class ChaAttnConvModule(nn.Module):
    def __init__(self, channels, k_size=7, mlp_ratio=4., groups=1):
        super().__init__()
        self.res = nn.Sequential(
            units.PDConv(channels=channels, k_size=k_size, groups=groups),
            units.ChannelAttention(),
            nn.LeakyReLU(inplace=True)
        )
        self.norm = nn.BatchNorm2d(channels)
        self.mlp = MLP(channels=channels, mlp_ratio=mlp_ratio)

    def forward(self, x, mask=None):
        if mask is not None:
            x = mask + x + self.res(x)
            x = mask + x + self.mlp(self.norm(x))
        elif mask is None:
            x = x + self.res(x)
            x = x + self.mlp(self.norm(x))

        return x


class MaskQueryTransModule(nn.Module):
    def __init__(self, channels, query_num, k_size, mlp_ratio=4., groups=1):
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.attn = self.GroupMaskQueryAttention(channels, query_num=query_num)
        self.mlp = MLP(channels=channels, mlp_ratio=mlp_ratio)
        self.sacb = self.MaskSpaAttnConvBlock(channels, k_size=k_size, groups=groups)

    class MaskSpaAttnConvBlock(nn.Module):
        def __init__(self, channels, k_size=7, groups=1):
            super().__init__()
            self.pdconv = units.PDConv(channels=channels, k_size=k_size, groups=groups)
            self.act = nn.LeakyReLU(inplace=True)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x, mask=None):
            if mask is not None:
                out = x + self.pdconv(x) * self.sigmoid(mask)
            elif mask is None:
                out = x + self.pdconv(x) * self.sigmoid(x)
            return self.act(out)

    class GroupMaskQueryAttention(nn.Module):
        def __init__(self, channels, query_num=8):
            super().__init__()
            self.query_num = query_num
            group_dim = channels // query_num
            self.scale = group_dim ** -0.5

            self.q = nn.Conv2d(channels, query_num, kernel_size=1, bias=False)
            self.kv = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False)
            self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

            self.reduce = nn.Conv2d(channels, channels, kernel_size=4, stride=4, bias=False)
            self.expand = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=4, bias=False)

        def forward(self, x, mask=None):
            x = self.reduce(x)
            B, C, H, W = x.shape
            N = H * W

            kv = self.kv(x).permute(0, 2, 3, 1).reshape(B, N, 2, C // self.query_num, self.query_num).permute(2, 0, 3,
                                                                                                              1, 4)
            k, v = kv.unbind(0)
            if mask is not None:
                mask = self.reduce(mask)
                q = x + mask
            elif mask is None:
                q = x
            q = self.q(q).permute(0, 2, 3, 1).reshape(B, N, 1, self.query_num).permute(0, 2, 1, 3)
            q = q.repeat(1, C // self.query_num, 1, 1)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = x.permute(0, 2, 1).reshape(B, C, H, W)

            x = self.expand(x)
            x = self.proj(x)
            return x

    def forward(self, x, mask=None):
        if mask is not None:
            x = self.sacb(x, mask) + self.attn(x, mask)
            x = mask + x + self.mlp(self.norm(x))
            return x

        elif mask is None:
            x = self.sacb(x) + self.attn(x)
            x = x + self.mlp(self.norm(x))
            return x


# ----------------------------------------------------------------------------------------------------------------------
class Stem(nn.Module):
    def __init__(self, channels, k_size, groups):
        super().__init__()
        self.head = nn.Conv2d(3, channels, kernel_size=k_size, padding=(k_size // 2), bias=False)
        self.mask = nn.Conv2d(1, channels, kernel_size=k_size, padding=(k_size // 2), bias=False)
        self.res = nn.Sequential(
            units.PDConv(channels=channels, k_size=k_size, groups=groups),
            units.ChannelAttention(),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x, mask=None):
        x = self.head(x)
        if mask is not None:
            mask = self.mask(mask)
            x = x + mask
            x = x + self.res(x)
            return x, mask
        elif mask is None:
            x = x + self.res(x)
            return x


class FusionEnDecoder(nn.Module):
    def __init__(self, ch_level, ch_level1, ch_level2, ch_level3, query_num, k_size=7, groups=1):
        super().__init__()
        self.cnt = nn.Sequential(
            ChaAttnConvModule(ch_level, k_size, groups),
            ChaAttnConvModule(ch_level, k_size, groups)
        )

        self.xdown = units.UpDownSample(ch_level, ch_level1, 0.5)
        self.xdown12 = units.UpDownSample(ch_level1, ch_level2, 0.5)
        self.xdown23 = units.UpDownSample(ch_level2, ch_level3, 0.5)
        self.mdown = units.UpDownSample(ch_level, ch_level1, 0.5)
        self.mdown12 = units.UpDownSample(ch_level1, ch_level2, 0.5)
        self.mdown23 = units.UpDownSample(ch_level2, ch_level3, 0.5)

        # encoder conv
        self.enc_c1 = ChaAttnConvModule(ch_level1, k_size, groups)
        self.enc_c2 = ChaAttnConvModule(ch_level2, k_size, groups)
        self.enc_c3 = ChaAttnConvModule(ch_level3, k_size, groups)
        self.enc_cdown12 = units.UpDownSample(ch_level1, ch_level2, 0.5)
        self.enc_cdown23 = units.UpDownSample(ch_level2, ch_level3, 0.5)

        # encoder trans
        self.enc_t1 = MaskQueryTransModule(ch_level1, k_size=k_size, query_num=query_num, groups=groups)
        self.enc_t2 = MaskQueryTransModule(ch_level2, k_size=k_size, query_num=query_num, groups=groups)
        self.enc_t3 = MaskQueryTransModule(ch_level3, k_size=k_size, query_num=query_num, groups=groups)
        self.enc_tdown12 = units.UpDownSample(ch_level1, ch_level2, 0.5)
        self.enc_tdown23 = units.UpDownSample(ch_level2, ch_level3, 0.5)

        self.cnxb1 = self.ConvNeXtBlock(ch_level1, k_size, groups)
        self.cnxb2 = self.ConvNeXtBlock(ch_level2, k_size, groups)
        self.cnxb3 = self.ConvNeXtBlock(ch_level3, k_size, groups)

        self.dec_c1 = ChaAttnConvModule(ch_level1, k_size, groups)
        self.dec_c2 = ChaAttnConvModule(ch_level2, k_size, groups)
        self.dec_c3 = ChaAttnConvModule(ch_level3, k_size, groups)
        self.dec_t1 = MaskQueryTransModule(ch_level1, k_size=k_size, query_num=query_num, groups=groups)
        self.dec_t2 = MaskQueryTransModule(ch_level2, k_size=k_size, query_num=query_num, groups=groups)
        self.dec_t3 = MaskQueryTransModule(ch_level3, k_size=k_size, query_num=query_num, groups=groups)

        self.up = units.UpDownSample(ch_level1, ch_level, 2)
        self.up21 = units.UpDownSample(ch_level2, ch_level1, 2)
        self.up32 = units.UpDownSample(ch_level3, ch_level2, 2)

    class ConvNeXtBlock(nn.Module):
        def __init__(self, channels, k_size, groups):
            super().__init__()
            self.res = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=k_size, padding=(k_size // 2), bias=False, groups=groups),
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=1, bias=False)
            )

        def forward(self, x):
            x = x + self.res(x)
            return x

    def convencoder(self, x, mask=None):
        x1 = self.xdown(x)
        x2 = self.xdown12(x1)
        x3 = self.xdown23(x2)

        if mask is not None:
            mask1 = self.mdown(mask)
            mask2 = self.mdown12(mask1)
            mask3 = self.mdown23(mask2)

            enc_c1 = self.enc_c1(x1, mask1)

            enc_c2 = self.enc_cdown12(enc_c1)
            enc_c2 = enc_c2 + x2
            enc_c2 = self.enc_c2(enc_c2, mask2)

            enc_c3 = self.enc_cdown23(enc_c2)
            enc_c3 = enc_c3 + x3
            enc_c3 = self.enc_c3(enc_c3, mask3)

        elif mask is None:
            enc_c1 = self.enc_c1(x1)

            enc_c2 = self.enc_cdown12(enc_c1)
            enc_c2 = enc_c2 + x2
            enc_c2 = self.enc_c2(enc_c2)

            enc_c3 = self.enc_cdown23(enc_c2)
            enc_c3 = enc_c3 + x3
            enc_c3 = self.enc_c3(enc_c3)

        return enc_c1, enc_c2, enc_c3

    def encoder_c(self, x, mask=None):
        H = x.size(2)
        W = x.size(3)

        x1_in_ltop = x[:, :, 0:int(H / 2), 0:int(W / 2)]
        x1_in_lbot = x[:, :, int(H / 2):H, 0:int(W / 2)]
        x1_in_rtop = x[:, :, 0:int(H / 2), int(W / 2):W]
        x1_in_rbot = x[:, :, int(H / 2):H, int(W / 2):W]

        if mask is not None:
            mask1_in_ltop = mask[:, :, 0:int(H / 2), 0:int(W / 2)]
            mask1_in_lbot = mask[:, :, int(H / 2):H, 0:int(W / 2)]
            mask1_in_rtop = mask[:, :, 0:int(H / 2), int(W / 2):W]
            mask1_in_rbot = mask[:, :, int(H / 2):H, int(W / 2):W]

            enc1_ltop, enc2_ltop, enc3_ltop = self.convencoder(x1_in_ltop, mask1_in_ltop)
            enc1_lbot, enc2_lbot, enc3_lbot = self.convencoder(x1_in_lbot, mask1_in_lbot)
            enc1_rtop, enc2_rtop, enc3_rtop = self.convencoder(x1_in_rtop, mask1_in_rtop)
            enc1_rbot, enc2_rbot, enc3_rbot = self.convencoder(x1_in_rbot, mask1_in_rbot)

        elif mask is None:
            enc1_ltop, enc2_ltop, enc3_ltop = self.convencoder(x1_in_ltop)
            enc1_lbot, enc2_lbot, enc3_lbot = self.convencoder(x1_in_lbot)
            enc1_rtop, enc2_rtop, enc3_rtop = self.convencoder(x1_in_rtop)
            enc1_rbot, enc2_rbot, enc3_rbot = self.convencoder(x1_in_rbot)

        enc1_left = torch.cat((enc1_ltop, enc1_lbot), 2)
        enc1_right = torch.cat((enc1_rtop, enc1_rbot), 2)
        enc_c1 = torch.cat((enc1_left, enc1_right), 3)
        enc2_left = torch.cat((enc2_ltop, enc2_lbot), 2)
        enc2_right = torch.cat((enc2_rtop, enc2_rbot), 2)
        enc_c2 = torch.cat((enc2_left, enc2_right), 3)
        enc3_left = torch.cat((enc3_ltop, enc3_lbot), 2)
        enc3_right = torch.cat((enc3_rtop, enc3_rbot), 2)
        enc_c3 = torch.cat((enc3_left, enc3_right), 3)

        return enc_c1, enc_c2, enc_c3

    def encoder_t(self, x, mask=None):
        # encoder trans
        x1 = self.xdown(x)
        x2 = self.xdown12(x1)
        x3 = self.xdown23(x2)

        if mask is not None:
            mask1, mask2, mask3 = mask

            enc_t1 = self.enc_t1(x1, mask1)

            enc_t2 = self.enc_tdown12(enc_t1)
            enc_t2 = enc_t2 + x2
            enc_t2 = self.enc_t2(enc_t2, mask2)

            enc_t3 = self.enc_tdown23(enc_t2)
            enc_t3 = enc_t3 + x3
            enc_t3 = self.enc_t3(enc_t3, mask3)

        elif mask is None:
            enc_t1 = self.enc_t1(x1)

            enc_t2 = self.enc_tdown12(enc_t1)
            enc_t2 = enc_t2 + x2
            enc_t2 = self.enc_t2(enc_t2)

            enc_t3 = self.enc_tdown23(enc_t2)
            enc_t3 = enc_t3 + x3
            enc_t3 = self.enc_t3(enc_t3)

        return enc_t1, enc_t2, enc_t3

    def decoder(self, x, mask=None):
        enc1, enc2, enc3 = x
        if mask is not None:
            mask1, mask2, mask3 = mask

            dec_t3 = self.dec_t3(enc3, mask3)
            dec_c3 = self.dec_c3(enc3, mask3)
            dec3 = dec_c3 + dec_t3

            dec2 = enc2 + self.up32(dec3)
            dec_t2 = self.dec_t2(dec2, mask2)
            dec_c2 = self.dec_c2(dec2, mask2)
            dec2 = dec_c2 + dec_t2

            dec1 = enc1 + self.up21(dec2)
            dec_t1 = self.dec_t1(dec1, mask1)
            dec_c1 = self.dec_c1(dec1, mask1)
        elif mask is None:
            dec_t3 = self.dec_t3(enc3)
            dec_c3 = self.dec_c3(enc3)
            dec3 = dec_c3 + dec_t3

            dec2 = enc2 + self.up32(dec3)
            dec_t2 = self.dec_t2(dec2)
            dec_c2 = self.dec_c2(dec2)
            dec2 = dec_c2 + dec_t2

            dec1 = enc1 + self.up21(dec2)
            dec_t1 = self.dec_t1(dec1)
            dec_c1 = self.dec_c1(dec1)

        dec1 = dec_c1 + dec_t1
        dec = self.up(dec1)

        return dec

    def forward(self, x, mask=None):
        if mask is not None:
            mask1 = self.mdown(mask)
            mask2 = self.mdown12(mask1)
            mask3 = self.mdown23(mask2)

            enc_c1, enc_c2, enc_c3 = self.encoder_c(x, mask)
            enc_t1, enc_t2, enc_t3 = self.encoder_t(x, (mask1, mask2, mask3))
            enc1 = enc_c1 + enc_t1
            enc2 = enc_c2 + enc_t2
            enc3 = enc_c3 + enc_t3

            enc1 = self.cnxb1(enc1)
            enc2 = self.cnxb2(enc2)
            enc3 = self.cnxb3(enc3)

            dec = self.cnt(x) + self.decoder((enc1, enc2, enc3), (mask1, mask2, mask3))
        elif mask is None:
            enc_c1, enc_c2, enc_c3 = self.encoder_c(x)
            enc_t1, enc_t2, enc_t3 = self.encoder_t(x)
            enc1 = enc_c1 + enc_t1
            enc2 = enc_c2 + enc_t2
            enc3 = enc_c3 + enc_t3

            enc1 = self.cnxb1(enc1)
            enc2 = self.cnxb2(enc2)
            enc3 = self.cnxb3(enc3)

            dec = self.cnt(x) + self.decoder((enc1, enc2, enc3))
        return dec


class PixelDetailEnhance(nn.Module):
    def __init__(self, ch_out, channels, k_size, groups):
        super().__init__()
        self.res = nn.Sequential(
            self.DetailBlock(channels, k_size, groups),
            self.DetailBlock(channels, k_size, groups),
            self.DetailBlock(channels, k_size, groups),
            nn.Conv2d(channels, ch_out, 1, bias=False)
        )

    class DetailBlock(nn.Module):
        def __init__(self, channels, k_size, groups):
            super().__init__()
            self.pdwconv = units.PDConv(channels, k_size=k_size, groups=groups)
            self.ca = units.ChannelAttention()
            self.act = nn.LeakyReLU(inplace=True)

        def forward(self, x):
            x = self.pdwconv(x)
            x = x + self.ca(x)
            x = self.act(x)
            return x

    def forward(self, x):
        return self.res(x)


class LMQFORMER(nn.Module):
    def __init__(self,
                 ch_img=3,
                 ch_unet=16,
                 channels=48,
                 query_num=8,
                 groups=1,
                 MASK=True
                 ):
        super().__init__()
        self.MASK = MASK
        ch_level = channels
        ch_level1 = channels
        ch_level2 = channels + ch_unet
        ch_level3 = channels + ch_unet * 2
        if 3 <= groups < 8:
            coder_groups = 4
        elif groups >= 8:
            coder_groups = 8
        else:
            coder_groups = groups

        self.VQVAE = GLP_VQVAE()

        self.Stem = Stem(channels=channels, k_size=3, groups=1)

        self.FusionEnDecoder = FusionEnDecoder(
            ch_level=ch_level, ch_level1=ch_level1, ch_level2=ch_level2, ch_level3=ch_level3,
            query_num=query_num, k_size=3, groups=coder_groups
        )
        self.PDE = PixelDetailEnhance(ch_out=ch_img, channels=channels, k_size=3, groups=1)

    def forward(self, x):
        if self.MASK:
            vaemask, _, _ = self.VQVAE(x)
            x_feat, mask = self.Stem(x, vaemask.detach())
            dec = self.FusionEnDecoder(x_feat, mask)
        elif not self.MASK:
            x_feat = self.Stem(x)
            dec = self.FusionEnDecoder(x_feat)

        res = self.PDE(dec)
        clean = x - res

        return clean, res


# ======================================================================================================================
if __name__ == '__main__':
    def model_complex(model, input_shape):
        macs, params = get_model_complexity_info(model, input_shape, as_strings=True,
                                                 print_per_layer_stat=False, verbose=True)
        print(f'====> Number of Model Params: {params}')
        print(f'====> Computational complexity: {macs}')


    model = LMQFORMER().cuda()
    model_complex(model, (3, 256, 256))




