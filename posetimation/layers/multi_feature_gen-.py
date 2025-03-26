
import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from posetimation.layers.SE_Model import SEAttention
from posetimation.layers.basic_model import ChainOfBasicBlocks
from posetimation.layers.basic_layer import conv_bn_relu
from torchvision.ops.deform_conv import DeformConv2d




class BasicFeatureMineBlock(nn.Module):
    """
    SEAtten + Deformable Atten
    """

    def __init__(self, channels=48, height=96, width=72):
        super(BasicFeatureMineBlock, self).__init__()

        self.c = channels
        self.h = height
        self.w = width

        # init mine
        self.se_atten = SEAttention(channel=self.c, expand=2)
        self.se_process = ChainOfBasicBlocks(self.c, self.c//8, num_blocks=1)

        n_kernel_group = 6
        n_offset_channel = 2 * 3 * 3 * n_kernel_group
        n_mask_channel = 3 * 3 * n_kernel_group

        # fine-grained mine
        self.dcn_offset = conv_bn_relu(self.c, n_offset_channel, 3, 1, padding=3, dilation=3, has_bn=False,
                                       has_relu=False)

        self.dcn_mask = conv_bn_relu(self.c, n_mask_channel, 3, 1, padding=3, dilation=3, has_bn=False, has_relu=False)

        self.dcn = DeformConv2d(self.c, self.c//8, 3, padding=3, dilation=3)

        self.pose_process = ChainOfBasicBlocks(self.c//8, self.c//8, num_blocks=2)

    def forward(self, x):
        _, C, H, W = x.shape
        # security check
        assert C == self.c and H == self.h and W == self.w, "Dimension Error!"

        x = self.se_atten(x)
        x = self.se_process(x)
        x_offset = self.dcn_offset(x)
        x_mask = self.dcn_mask(x)
        x = self.dcn(x, x_offset, x_mask)
        x = self.pose_process(x)

        return x


class ChainofFMBs(nn.Module):
    def __init__(self, channels=48, height=96, width=72, num_layers=2):
        super(ChainofFMBs, self).__init__()
        self.num_layers = num_layers

        block_list = []
        for i in range(self.num_layers):
            block_list.append(BasicFeatureMineBlock(channels, height, width))
        self.layers = nn.ModuleList(block_list)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MultiBranchGenerator(nn.Module):
    def __init__(self, num_branch=3, c=48, h=96, w=72, num_layers=2):
        super(MultiBranchGenerator, self).__init__()
        self.num_branch = num_branch
        branches = []
        for i in range(self.num_branch):
            branches.append(ChainofFMBs(channels=c, height=h, width=w, num_layers=num_layers))
        self.branches = nn.ModuleList(branches)

    def forward(self, x):
        results = []
        for idx in range(len(self.branches)):
            results.append(self.branches[idx](x))
        return results

