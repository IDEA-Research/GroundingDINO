import torch
import math
import random
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch import nn

torch_ver = torch.__version__[:3]

__all__ = ['BatchDrop', 'BatchFeatureErase_Top', 'BatchRandomErasing',
           'PAM_Module', 'CAM_Module', 'Dual_Module', 'SE_Module']


class BatchRandomErasing(nn.Module):
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        super(BatchRandomErasing, self).__init__()

        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def forward(self, img):
        if self.training:

            if random.uniform(0, 1) > self.probability:
                return img

            for attempt in range(100):

                area = img.size()[2] * img.size()[3]

                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1 / self.r1)

                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))

                if w < img.size()[3] and h < img.size()[2]:
                    x1 = random.randint(0, img.size()[2] - h)
                    y1 = random.randint(0, img.size()[3] - w)
                    if img.size()[1] == 3:
                        img[:, 0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                        img[:, 1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                        img[:, 2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                    else:
                        img[:, 0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    return img

        return img


class BatchDrop(nn.Module):
    """
    Ref: Batch DropBlock Network for Person Re-identification and Beyond
    https://github.com/daizuozhuo/batch-dropblock-network/blob/master/models/networks.py
    Created by: daizuozhuo
    """

    def __init__(self, h_ratio, w_ratio):
        super(BatchDrop, self).__init__()
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio

    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rh = round(self.h_ratio * h)
            rw = round(self.w_ratio * w)
            sx = random.randint(0, h - rh)
            sy = random.randint(0, w - rw)
            mask = x.new_ones(x.size())
            mask[:, :, sx:sx + rh, sy:sy + rw] = 0
            x = x * mask
        return x


class BatchDropTop(nn.Module):
    """
    Ref: Top-DB-Net: Top DropBlock for Activation Enhancement in Person Re-Identification
    https://github.com/RQuispeC/top-dropblock/blob/master/torchreid/models/bdnet.py
    Created by: RQuispeC

    """

    def __init__(self, h_ratio):
        super(BatchDropTop, self).__init__()
        self.h_ratio = h_ratio

    def forward(self, x, visdrop=False):
        if self.training or visdrop:
            b, c, h, w = x.size()
            rh = round(self.h_ratio * h)
            act = (x**2).sum(1)
            act = act.view(b, h * w)
            act = F.normalize(act, p=2, dim=1)
            act = act.view(b, h, w)
            max_act, _ = act.max(2)
            ind = torch.argsort(max_act, 1)
            ind = ind[:, -rh:]
            mask = []
            for i in range(b):
                rmask = torch.ones(h)
                rmask[ind[i]] = 0
                mask.append(rmask.unsqueeze(0))
            mask = torch.cat(mask)
            mask = torch.repeat_interleave(mask, w, 1).view(b, h, w)
            mask = torch.repeat_interleave(mask, c, 0).view(b, c, h, w)
            if x.is_cuda:
                mask = mask.cuda()
            if visdrop:
                return mask
            x = x * mask
        return x


class BatchFeatureErase_Top(nn.Module):
    """
    Ref: Top-DB-Net: Top DropBlock for Activation Enhancement in Person Re-Identification
    https://github.com/RQuispeC/top-dropblock/blob/master/torchreid/models/bdnet.py
    Created by: RQuispeC

    """

    def __init__(self, channels, bottleneck_type, h_ratio=0.33, w_ratio=1., double_bottleneck=False):
        super(BatchFeatureErase_Top, self).__init__()

        self.drop_batch_bottleneck = bottleneck_type(channels, 512)

        self.drop_batch_drop_basic = BatchDrop(h_ratio, w_ratio)
        self.drop_batch_drop_top = BatchDropTop(h_ratio)

    def forward(self, x, drop_top=True, bottleneck_features=True, visdrop=False):
        features = self.drop_batch_bottleneck(x)

        if drop_top:
            x = self.drop_batch_drop_top(features, visdrop=visdrop)
        else:
            x = self.drop_batch_drop_basic(features, visdrop=visdrop)
        if visdrop:
            return x  # x is dropmask
        if bottleneck_features:
            return x, features
        else:
            return x


class SE_Module(Module):

    def __init__(self, channels, reduction=4):
        super(SE_Module, self).__init__()
        self.fc1 = Conv2d(channels, channels // reduction,
                          kernel_size=1, padding=0)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels,
                          kernel_size=1, padding=0)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class PAM_Module(Module):
    """ Position attention module"""
    # Ref from SAGAN

    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(
            m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class CAM_Module(Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        # proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1).contiguous()
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(
            energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class Dual_Module(Module):
    """
    # Created by: CASIA IVA
    # Email: jliu@nlpr.ia.ac.cn
    # Copyright (c) 2018

    # Reference: Dual Attention Network for Scene Segmentation
    # https://arxiv.org/pdf/1809.02983.pdf
    # https://github.com/junfu1115/DANet/blob/master/encoding/nn/attention.py
    """

    def __init__(self, in_dim):
        super(Dual_Module).__init__()
        self.indim = in_dim
        self.pam = PAM_Module(in_dim)
        self.cam = CAM_Module(in_dim)

    def forward(self, x):
        out1 = self.pam(x)
        out2 = self.cam(x)
        return out1 + out2