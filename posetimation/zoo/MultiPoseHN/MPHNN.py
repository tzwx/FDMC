#!/usr/bin/python
# -*- coding:utf8 -*-

import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from einops import rearrange
from engine.defaults.constant import MODEL_REGISTRY
from posetimation.layers.basic_layer import conv_bn_relu
from posetimation.layers.basic_model import ChainOfBasicBlocks, ChainOfBasicBlocksFix
from posetimation.backbones.hrnet import HRNetPlus
from posetimation.layers.multi_feature_gen import MultiBranchGenerator
from torchvision.ops.deform_conv import DeformConv2d
from engine.defaults import TRAIN_PHASE
from posetimation.layers.SE_Model import SEAttention

__all__ = ["MPHNNJ"]
BN_MOMENTUM = 0.1

import logging
import os.path as osp
import torch
from torch.nn.functional import kl_div
from torch.nn import init


@MODEL_REGISTRY.register()
class MPHNNJ(nn.Module):
    """

    """

    @classmethod
    def get_model_hyper_parameters(cls, cfg):
        bbox_enlarge_factor = cfg.DATASET.BBOX_ENLARGE_FACTOR
        rot_factor = cfg.TRAIN.ROT_FACTOR
        SCALE_FACTOR = cfg.TRAIN.SCALE_FACTOR

        if not isinstance(SCALE_FACTOR, list):
            temp = SCALE_FACTOR
            SCALE_FACTOR = [SCALE_FACTOR, SCALE_FACTOR]
        scale_bottom = 1 - SCALE_FACTOR[0]
        scale_top = 1 + SCALE_FACTOR[1]

        paramer = "bbox_{}_rot_{}_scale_{}-{}".format(bbox_enlarge_factor, rot_factor, scale_bottom,
                                                      scale_top)

        if cfg.LOSS.HEATMAP_MSE.USE:
            paramer += f"_MseLoss_{cfg.LOSS.HEATMAP_MSE.WEIGHT}"

        return paramer

    def __init__(self, cfg, is_train, **kwargs):
        super(MPHNNJ, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.pretrained = cfg.MODEL.PRETRAINED
        self.is_train = is_train
        if self.is_train == TRAIN_PHASE:
            self.is_train = True
        else:
            self.is_train = False
        self.pretrained_layers = ['*']
        self.hrnet = HRNetPlus(cfg, self.is_train)
        self.freeze_hrnet_weight = cfg['MODEL']["FREEZE_HRNET_WEIGHTS"]

        # Basic Feature Extraction
        self.all_agg_block = ChainOfBasicBlocks(input_channel=256 * 5, ouput_channel=256, num_blocks=3)

        # Multi Branch
        self.num_branch = 4
        self.multi_branch_gen = MultiBranchGenerator(num_branch=self.num_branch, c=256)
        self.global_fusion = ChainOfBasicBlocks(32 * self.num_branch, 32)

        self.patch_embed_hm = PatchEmbed((96, 72), (16, 12), 17, 768)
        self.patch_embed = PatchEmbed((96, 72), (16, 12), 32, 768)

        self.pose_feature = nn.Sequential(Block(dim=768, num_heads=8))
        self.query_pose = nn.Linear(768, 768)

        self.kv = nn.ModuleList([
            nn.Linear(768, 768 * 2),
            nn.Linear(768, 768 * 2),
            nn.Linear(768, 768 * 2),
            nn.Linear(768, 768 * 2),
            nn.Linear(768, 768 * 2)
        ])

        self.interaction = nn.ModuleList([
            CrossAttenBlock(768),
            CrossAttenBlock(768),
            CrossAttenBlock(768),
            CrossAttenBlock(768),
            CrossAttenBlock(768)
        ])

        sub_heatmap_heads = []
        for i in range(self.num_branch):
            sub_heatmap_heads.append(nn.Conv2d(768, 17, 3, 1, 1))
        self.sub_heatmap_heads = nn.ModuleList(sub_heatmap_heads)

        self.global_heatmap_head = nn.Conv2d(32, 17, 3, 1, 1)

        self.heatmap_fusion = ChainOfBasicBlocks(17 * (self.num_branch ), 17, num_blocks=1)
        self.softmax = torch.nn.Softmax(dim=1)

        self.init_weights()

        if self.freeze_hrnet_weight:
            self.hrnet.freeze_weight()

    def forward(self, kf_x, sup_x, **kwargs):
        """
        kf_x:  [batch, 3, 384, 288]
        sup_x: [batch, 3 * num, 384, 288]]
        """

        batch_size, num_sup = kf_x.shape[0], sup_x.shape[1] // 3
        sup_x = torch.cat(torch.chunk(sup_x, num_sup, dim=1), dim=0)  # [batch * num, 3, 384, 288]]

        x = torch.cat([kf_x, sup_x], dim=0)
        x_bb_hm, x_bb_feat = self.hrnet(x, multi_scale=True)

        x_bb_feat = self.dim_mapping(x_bb_feat)

        x_bb_hm_list = torch.chunk(x_bb_hm, num_sup + 1, dim=0)

        x_bb_feat_list = torch.chunk(x_bb_feat[-1], num_sup + 1, dim=0)
        kf_bb_hm, kf_bb_feat = x_bb_hm_list[0], x_bb_feat_list[0]
        sup_bb_hm_list, sup_bb_feat_list = x_bb_hm_list[1:], x_bb_feat_list[1:]

        # Base video feature extraction
        supp_agg_f = torch.cat(sup_bb_feat_list, dim=1)
        f_fusion = self.all_agg_block(supp_agg_f)

        supp_agg_t = torch.stack(sup_bb_feat_list, dim=1)
        t_fusion = torch.mean(supp_agg_t, dim=1)

        basic_feat = f_fusion + t_fusion

        # Generate multi-sub-features Coarse-to-Fine
        multi_branch_feat_list = self.multi_branch_gen(basic_feat)

        x1, x2, x3, x4 = multi_branch_feat_list
        global_feat = self.global_fusion(torch.cat(multi_branch_feat_list, dim=1))

        multi_branch_feat_list.append(global_feat)

        for feat in multi_branch_feat_list:
            feat = self.patch_embed(feat)
        hm = self.patch_embed_hm(x_bb_hm)
        hm_f = self.pose_feature(hm)

        Q_hm = self.query_pose(hm_f)

        heatmaps = []
        for idx in range(len(multi_branch_feat_list)):
            kv = self.kv(multi_branch_feat_list[idx])
            k, v = torch.chunk(kv, 2, -1)
            q_feat = self.interaction[idx](Q_hm, k, v)
            q_feat = q_feat.permute(0, 3, 1, 2)
            heatmaps.append(self.sub_heatmap_heads[idx](q_feat))

        final_heatmaps = self.heatmap_fusion(torch.cat(heatmaps, dim=1))

        if self.is_train:

            mi_fb_y = self.feat_label_mi_estimation(basic_feat, final_heatmaps, motion=False)
            mi_fb_x1 = self.feat_feat_mi_estimation(basic_feat, x1)
            mi_fb_x2 = self.feat_feat_mi_estimation(basic_feat, x2)
            mi_fb_x3 = self.feat_feat_mi_estimation(basic_feat, x3)
            mi_fb_x4 = self.feat_feat_mi_estimation(basic_feat, x4)
            mi_fb_xg = self.feat_feat_mi_estimation(basic_feat, global_feat)

            disentangle_loss = 0.
            out = multi_branch_feat_list[:-1]
            temperature = 0.05
            cnt_min = 0
            num_rec = [0] * self.num_branch
            for j in range(1, self.num_branch):
                for i in range(self.num_branch):
                    if num_rec[i] < self.num_branch - 1 and num_rec[(i + j) % self.num_branch] < self.num_branch - 1:
                        disentangle_loss += kl_div(self.softmax(out[i] / temperature),
                                                   self.softmax(out[(i + j) % self.num_branch] / temperature),
                                                   reduction='mean')
                        num_rec[i] += 1
                        num_rec[(i + j) % self.num_branch] += 1
                        cnt_min += 1
                    if sum(num_rec) == (self.num_branch - 1) * self.num_branch:
                        break
                if sum(num_rec) == (self.num_branch - 1) * self.num_branch:
                    break

            # debug = False
            # if debug:
            #     return final_heatmaps, kf_bb_hm, heatmaps, [], [], []
            # else:
            return final_heatmaps, kf_bb_hm, heatmaps, \
                   [mi_fb_y, mi_fb_x1, mi_fb_x2, mi_fb_x3, mi_fb_x4, mi_fb_xg], \
                   disentangle_loss
        else:
            return final_heatmaps, kf_bb_hm

    def init_weights(self, *args, **kwargs):
        logger = logging.getLogger(__name__)
        hrnet_name_set = set()
        for module_name, module in self.named_modules():
            # rough_pose_estimation_net 单独判断一下
            if module_name.split('.')[0] == "hrnet":
                hrnet_name_set.add(module_name)

            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.001)
                for name, _ in module.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight, std=0.001)

                for name, _ in module.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(module.bias, 0)
            else:
                for name, _ in module.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(module.bias, 0)
                    if name in ['weights']:
                        nn.init.normal_(module.weight, std=0.001)

        if osp.isfile(self.pretrained):
            pretrained_state_dict = torch.load(self.pretrained)
            if 'state_dict' in pretrained_state_dict.keys():
                pretrained_state_dict = pretrained_state_dict['state_dict']
            logger.info('{} => loading pretrained model {}'.format(self.__class__.__name__,
                                                                   self.pretrained))

            if list(pretrained_state_dict.keys())[0].startswith('module.'):
                model_state_dict = {k[7:]: v for k, v in pretrained_state_dict.items()}
            else:
                model_state_dict = pretrained_state_dict

            need_init_state_dict = {}
            for name, m in model_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers or self.pretrained_layers[0] is '*':
                    layer_name = name.split('.')[0]
                    if layer_name in hrnet_name_set:
                        need_init_state_dict[name] = m
                    else:
                        # 为了适应原本hrnet得预训练网络
                        new_layer_name = "hrnet.{}".format(layer_name)
                        if new_layer_name in hrnet_name_set:
                            parameter_name = "hrnet.{}".format(name)
                            need_init_state_dict[parameter_name] = m

            self.load_state_dict(need_init_state_dict, strict=False)
            # self.load_state_dict(need_init_state_dict, strict=False)
        elif self.pretrained:
            # raise NotImplementedError
            logger.error('=> please download pre-trained models first!')

        # self.freeze_weight()

        self.logger.info("Finish init_weights")

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        # print(classname)
        if classname.find('Conv') != -1:
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif classname.find('Linear') != -1:
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
            init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm1d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    def feat_label_mi_estimation(self, Feat, Y, motion=False):
        batch_size = Feat.shape[0]
        temperature = 0.05
        pred_Y = self.hrnet.final_layer(Feat)  # B,48,96,72 -> B,17,96,72
        pred_Y = pred_Y.reshape(batch_size, 17, -1).reshape(batch_size * 17, -1)
        Y = Y.reshape(batch_size, 17, -1).reshape(batch_size * 17, -1)
        if motion:
            motion_Y = self.motion_heatmap_head(Feat)
            mi = kl_div(input=self.softmax(motion_Y.detach() / temperature),
                        target=self.softmax(Y / temperature),
                        reduction='mean')  # pixel-level
        else:
            mi = kl_div(input=self.softmax(pred_Y.detach() / temperature),
                        target=self.softmax(Y / temperature),
                        reduction='mean')  # pixel-level

        return mi

    def feat_feat_mi_estimation(self, F1, F2, freeze=True):

        batch_size = F1.shape[0]
        temperature = 0.05
        F1 = F1.reshape(batch_size, 48, -1).reshape(batch_size * 48, -1)
        F2 = F2.reshape(batch_size, 48, -1).reshape(batch_size * 48, -1)
        if freeze:
            mi = kl_div(input=self.softmax(F1.detach() / temperature),
                        target=self.softmax(F2 / temperature))
        else:
            mi = kl_div(input=self.softmax(F1 / temperature),
                        target=self.softmax(F2 / temperature))

        return mi


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
        self.se_process = ChainOfBasicBlocks(48, 48, num_blocks=1)
        self.pose_process = self.se_process.cuda()

        n_kernel_group = 6
        n_offset_channel = 2 * 3 * 3 * n_kernel_group
        n_mask_channel = 3 * 3 * n_kernel_group

        # fine-grained mine
        self.dcn_offset = conv_bn_relu(48, n_offset_channel, 3, 1, padding=3, dilation=3, has_bn=False,
                                       has_relu=False)
        self.dcn_offset = self.dcn_offset.cuda()

        self.dcn_mask = conv_bn_relu(48, n_mask_channel, 3, 1, padding=3, dilation=3, has_bn=False, has_relu=False)
        self.dcn_mask = self.dcn_mask.cuda()

        self.dcn = DeformConv2d(48, 48, 3, padding=3, dilation=3)
        self.dcn = self.dcn.cuda()

        self.pose_process = ChainOfBasicBlocks(48, 48, num_blocks=2)
        self.pose_process = self.pose_process.cuda()

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


class CrossAttention(nn.Module):

    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, q, k, v):
        q = self.norm(q)
        k = self.norm(k)
        v = self.norm(v)

        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads),
            (q, k, v))

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class CrossAttenBlock(nn.Module):

    def __init__(self, dim, num_heads=8, head_dim=64, drop_path=0., mlp_ratio=4., act_layer=nn.GELU):
        super().__init__()

        self.attn = CrossAttention(dim, heads=num_heads, dim_head=head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_path)

    def forward(self, q, k, v):
        q = q + self.drop_path(self.attn(q, k, v))
        q = q + self.drop_path(self.mlp(q))
        return q


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, ratio=1):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (ratio ** 2)
        self.patch_shape = (int(img_size[0] // patch_size[0] * ratio), int(img_size[1] // patch_size[1] * ratio))
        self.origin_patch_shape = (int(img_size[0] // patch_size[0]), int(img_size[1] // patch_size[1]))
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=(patch_size[0] // ratio),
                              padding=4 + 2 * (ratio // 2 - 1))

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.proj(x)  # [batch, 1280, 16, 12]
        Hp, Wp = x.shape[2], x.shape[3]

        x = x.flatten(2).transpose(1, 2)  # [64, 16*12, 1280] -> [batch, H*W, Channel]
        return x, (Hp, Wp)


class DropPath(nn.Module):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None, ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim

        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)  # [3, batch, num_head, patch, channels]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, attn_head_dim=None
                 ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim
        )

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
