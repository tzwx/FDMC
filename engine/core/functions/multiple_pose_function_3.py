#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: Haoming Chen
    E-mail: chenhaomingbob@163.com
    Time: 2020/09/27
    Description:
"""
import logging
import os.path as osp
import sys
import time

import numpy as np
import random
import torch
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

from datasets.process import get_final_preds, dark_get_final_preds
from datasets.transforms import reverse_transforms
# from engine.core import AverageMeter
from engine.core.base import BaseFunction, AverageMeter
from engine.core.utils.evaluate import accuracy
# from engine.core.base import
# from engine.evaludate import accuracy
from engine.defaults import VAL_PHASE, TEST_PHASE, TRAIN_PHASE
from engine.defaults.constant import CORE_FUNCTION_REGISTRY
from posetimation.loss.cosine_loss import StructureCosineSimilarity
from posetimation.loss.integral_loss import IntegralL1Loss
from posetimation.loss.mse_loss import JointMSELoss, JointSequenceMSELoss, OHKMJointMSELoss
from utils.utils_bbox import cs2box
from utils.utils_folder import create_folder
from utils.utils_image_tensor import tensor2im

from tqdm.auto import tqdm

from tabulate import tabulate
from termcolor import colored
from datasets.process.pose_process import flip_back

import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)


# np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


@CORE_FUNCTION_REGISTRY.register()
class MFunctionMI3(BaseFunction):

    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.output_dir = cfg.OUTPUT_DIR

        if "criterion" in kwargs.keys():
            self.criterion = kwargs["criterion"]
        if "tb_log_dir" in kwargs.keys():
            self.tb_log_dir = kwargs["tb_log_dir"]
        if "writer_dict" in kwargs.keys():
            self.writer_dict = kwargs["writer_dict"]

        ##
        self.PE_Name = kwargs.get("PE_Name", "DCPOSE")
        self.max_iter_num = 0
        self.dataloader_iter = None
        self.tb_writer = None
        self.global_steps = 0

        # self.HeatmapMSELOSS_criterion = JointMSELoss()
        self.SeqHeatmapMSELOSS_criterion = JointSequenceMSELoss()
        self.HeatmapMSELOSS_criterion = JointMSELoss()
        self.OHKMHeatmapMSELOSS_criterion = OHKMJointMSELoss()
        self.IntegralL1Loss_criterion = IntegralL1Loss()
        self.StructureCosineSimilarityLoss_criterion = StructureCosineSimilarity()

        self.use_mse_loss = self.cfg.LOSS.HEATMAP_MSE.USE
        self.use_integral_l1_loss = self.cfg.LOSS.INTEGRAL_L1.USE
        self.use_structure_cosine_loss = self.cfg.LOSS.STRUCTURE_COSINE.USE

        self.mse_weight = self.cfg.LOSS.HEATMAP_MSE.WEIGHT
        self.integral_l1_weight = self.cfg.LOSS.INTEGRAL_L1.WEIGHT
        self.structure_cosine_weight = self.cfg.LOSS.STRUCTURE_COSINE.WEIGHT

    def train(self, model, epoch, optimizer, dataloader, tb_writer_dict, **kwargs):
        self.tb_writer = tb_writer_dict["writer"]
        self.global_steps = tb_writer_dict["global_steps"]
        logger = logging.getLogger(__name__)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_mi = AverageMeter()
        losses_mi_sufficient, losses_mi_diverse = [AverageMeter() for _ in range(2)]
        acc = AverageMeter()
        acc_kf_backbone = AverageMeter()
        model.train()
        self.use_coefficient = False
        self.use_KLloss = False
        self.new_citeration = torch.nn.KLDivLoss()
        self.max_iter_num = len(dataloader)
        self.dataloader_iter = iter(dataloader)
        end = time.time()
        # acc_sup1, acc_sup2, acc_sup3, acc_sup4 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        losses_mse = AverageMeter()

        for iter_step in tqdm(list(range(self.max_iter_num))):

            input_x, sup_x, target_heatmaps, target_heatmaps_weight, meta = next(
                self.dataloader_iter)
            batch_size = input_x.size(0)
            num_sup = sup_x.size(1) // 3
            data_time.update(time.time() - end)

            target_heatmaps = target_heatmaps.cuda(non_blocking=True)
            target_heatmaps_weight = target_heatmaps_weight.cuda(non_blocking=True)

            pred_heatmaps, hrnet_heatmaps, intermediate_hm, sufficient_loss_list, diverse_loss = model(input_x.cuda(),
                                                                                                       sup_x.cuda(),
                                                                                                       iter=iter_step)

            loss = 0.
            loss_heatmaps = self.HeatmapMSELOSS_criterion(pred_heatmaps, target_heatmaps, target_heatmaps_weight)
            losses_mse.update(loss_heatmaps.item(), batch_size)
            loss += loss_heatmaps

            # if len(sufficient_loss_list) > 0:
            loss_mi, mi_weight = 0., .1

            mi_fb_y, mi_fb_x1, mi_fb_x2, mi_fb_x3, mi_fb_x4, mi_fb_xg = sufficient_loss_list
            sufficient_loss = mi_fb_y - mi_fb_x1 + mi_fb_y - mi_fb_x2 + mi_fb_y - mi_fb_x3 \
                              + mi_fb_y - mi_fb_x4 + mi_fb_y - mi_fb_xg
            losses_mi_sufficient.update(sufficient_loss.item(), batch_size * num_sup)

            loss_mi += sufficient_loss * mi_weight
            losses_mi.update(loss_mi.item(), batch_size * num_sup)

            loss += loss_mi

            loss_disentanglement, kl_weight = 0., .01
            loss_disentanglement += diverse_loss * kl_weight
            losses_mi_diverse.update(loss_disentanglement.item(), batch_size * num_sup)

            loss += loss_disentanglement
            losses.update(loss.item(), batch_size)
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            #

            _, avg_acc, cnt1, _ = accuracy(pred_heatmaps.detach().cpu().numpy(),
                                           target_heatmaps.detach().cpu().numpy())
            acc.update(avg_acc, cnt1)

            _, kf_bb_hm_acc, cnt1, _ = accuracy(hrnet_heatmaps.detach().cpu().numpy(),
                                                target_heatmaps.detach().cpu().numpy())
            acc_kf_backbone.update(kf_bb_hm_acc, cnt1)

            batch_time.update(time.time() - end)
            end = time.time()
            if iter_step % self.cfg.PRINT_FREQ == 0 or iter_step >= self.max_iter_num - 1:
                msg = f'Epoch: [{epoch}][{iter_step}/{self.max_iter_num}]\t' \
                      f'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      f'Speed {input_x.size(0) / batch_time.val:.1f} samples/s\t' \
                      f'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\n'

                # loss_table_header = ["Loss"]
                # loss_table_data = [f"{losses.val:.5f} ({losses.avg:.5f})"]

                loss_table_header = ["Loss", "Loss_mse", "Loss_MI",
                                     "Sufficient_Loss_MI", "Diverse_Loss_MI"]
                loss_table_data = [f"{losses.val:.5f} ({losses.avg:.5f})"]
                loss_table_data.extend([f"{losses_mse.val:.5f} ({losses_mse.avg:.5f})"])
                loss_table_data.extend([f"{losses_mi.val:.5f} ({losses_mi.avg:.5f})"])
                loss_table_data.extend([f"{losses_mi_sufficient.val:.5f} ({losses_mi_sufficient.avg:.5f})"])
                loss_table_data.extend([f"{losses_mi_diverse.val:.5f} ({losses_mi_diverse.avg:.5f})"])


                acc_table_header = ['Final', 'H_Kf_backbone']
                acc_table_data = [f'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t']
                acc_table_data.extend([f'{acc_kf_backbone.val:.3f} ({acc_kf_backbone.avg:.3f})'])


                loss_table = tabulate([loss_table_data], tablefmt='pipe', headers=loss_table_header,
                                      numalign='center')
                acc_table = tabulate([acc_table_data], tablefmt='pipe', headers=acc_table_header,
                                     numalign='center')
                msg += "=> Loss Table: \n" + colored(loss_table, "yellow") + '\n'
                msg += "=> Acc Table: \n" + colored(acc_table, "yellow")

                logger.info(msg)

                # For Tensorboard
            self.tb_writer.add_scalar('train_loss', losses.val, self.global_steps)
            self.tb_writer.add_scalar('train_acc', acc.val, self.global_steps)
            for i in range(len(optimizer.state_dict()['param_groups'])):
                self.tb_writer.add_scalar(f'Params_Group_{i}_learning_rate',
                                          optimizer.state_dict()['param_groups'][i]['lr'], epoch)
                self.global_steps += 1

        self.tb_writer.add_scalar('train_acc_avg', acc.avg, epoch)
        tb_writer_dict["global_steps"] = self.global_steps

    def eval(self, model, dataloader, tb_writer_dict, **kwargs):
        logger = logging.getLogger(__name__)

        self.tb_writer = tb_writer_dict["writer"]
        self.global_steps = tb_writer_dict["global_steps"]

        batch_time, data_time, losses, acc = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        acc_kf_backbone = AverageMeter()
        phase = kwargs.get("phase", VAL_PHASE)
        epoch = kwargs.get("epoch", "specified_model")
        # switch to evaluate mode
        model.eval()

        self.max_iter_num = len(dataloader)
        self.dataloader_iter = iter(dataloader)
        dataset = dataloader.dataset
        # prepare data fro validate
        num_samples = len(dataset)
        all_preds = np.zeros((num_samples, self.cfg.MODEL.NUM_JOINTS, 3), dtype=np.float)
        all_bb = np.zeros((num_samples, self.cfg.MODEL.NUM_JOINTS, 3), dtype=np.float)
        all_boxes = np.zeros((num_samples, 6))
        image_path = []
        filenames = []
        filenames_map = {}
        filenames_counter = 0
        imgnums = []
        idx = 0
        acc_threshold = 0.7
        ##
        assert phase in [VAL_PHASE, TEST_PHASE]
        if phase == VAL_PHASE:
            FLIP_TEST = self.cfg.VAL.FLIP
            SHIFT_HEATMAP = True
        elif phase == TEST_PHASE:
            FLIP_TEST = self.cfg.TEST.FLIP
            SHIFT_HEATMAP = True
        ###
        result_output_dir, vis_output_dir = self.vis_info(logger, phase, epoch)
        ###
        logger.info(
            "PHASE:{}, FLIP_TEST:{}, SHIFT_HEATMAP:{}".format(phase, FLIP_TEST, SHIFT_HEATMAP))
        with torch.no_grad():
            end = time.time()
            num_batch = len(dataloader)
            for iter_step in tqdm(list(range(self.max_iter_num))):
                key_frame_input, sup_frame_input, target_heatmaps, target_heatmaps_weight, meta = next(
                    self.dataloader_iter)

                # if iter_step < 4:
                #     print("iter_step: ", iter_step)
                #     continue
                # if iter_step > 80:
                #     sys.exit("hello end!")

                data_time.update(time.time() - end)
                target_heatmaps = target_heatmaps.cuda(non_blocking=True)
                pred_heatmaps, kf_bb_hm = model(key_frame_input.cuda(), sup_frame_input.cuda(), iter=iter_step)
                # FLIP_TEST = True
                if FLIP_TEST:
                    input_key_flipped = key_frame_input.flip(3)
                    input_sup_flipped = sup_frame_input.flip(3)

                    pred_heatmaps_flipped, kf_bb_hm_flipped = model(input_key_flipped.cuda(),
                                                                    input_sup_flipped.cuda())

                    pred_heatmaps_flipped = flip_back(pred_heatmaps_flipped.cpu().numpy(),
                                                      dataset.flip_pairs)
                    kf_bb_hm_flipped = flip_back(kf_bb_hm_flipped.cpu().numpy(), dataset.flip_pairs)

                    pred_heatmaps_flipped = torch.from_numpy(pred_heatmaps_flipped.copy()).cuda()
                    kf_bb_hm_flipped = torch.from_numpy(kf_bb_hm_flipped.copy()).cuda()

                    if SHIFT_HEATMAP:
                        pred_heatmaps_flipped[:, :, :, 1:] = pred_heatmaps_flipped.clone()[:, :, :,
                                                             0:-1]
                        kf_bb_hm_flipped[:, :, :, 1:] = kf_bb_hm_flipped.clone()[:, :, :, 0:-1]
                    pred_heatmaps = (pred_heatmaps + pred_heatmaps_flipped) * 0.5
                    kf_bb_hm = (kf_bb_hm + kf_bb_hm_flipped) * 0.5

                _, avg_acc, cnt, _ = accuracy(pred_heatmaps.detach().cpu().numpy(),
                                              target_heatmaps.detach().cpu().numpy())
                acc.update(avg_acc, cnt)

                _, kf_bb_hm_acc, cnt1, _ = accuracy(kf_bb_hm.detach().cpu().numpy(),
                                                    target_heatmaps.detach().cpu().numpy())
                acc_kf_backbone.update(kf_bb_hm_acc, cnt1)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if iter_step % self.cfg.PRINT_FREQ == 0 or iter_step >= (num_batch - 1):
                    msg = 'Val: [{0}/{1}]\t' \
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                          'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                          'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(iter_step, num_batch,
                                                                          batch_time=batch_time,
                                                                          data_time=data_time,
                                                                          acc=acc)
                    logger.info(msg)

                #### for eval ####
                for ff in range(len(meta['image'])):
                    cur_nm = meta['image'][ff]
                    if not cur_nm in filenames_map:
                        filenames_map[cur_nm] = [filenames_counter]
                    else:
                        filenames_map[cur_nm].append(filenames_counter)
                    filenames_counter += 1

                center = meta['center'].numpy()
                scale = meta['scale'].numpy()
                score = meta['score'].numpy()
                num_images = key_frame_input.size(0)

                pred_coord, our_maxvals = dark_get_final_preds(pred_heatmaps.clone().cpu().numpy(),
                                                               center, scale)
                all_preds[idx:idx + num_images, :, :2] = pred_coord
                all_preds[idx:idx + num_images, :, 2:3] = our_maxvals

                bb_coord, bb_maxvals = dark_get_final_preds(kf_bb_hm.clone().cpu().numpy(), center,
                                                            scale)
                all_bb[idx:idx + num_images, :, :2] = bb_coord
                all_bb[idx:idx + num_images, :, 2:3] = bb_maxvals

                all_boxes[idx:idx + num_images, 0:2] = center[:, 0:2]
                all_boxes[idx:idx + num_images, 2:4] = scale[:, 0:2]
                all_boxes[idx:idx + num_images, 4] = np.prod(scale * 200, 1)
                all_boxes[idx:idx + num_images, 5] = score

                image_path.extend(meta['image'])
                idx += num_images

                self.global_steps += 1

                self.vis_hook(meta["image"], pred_coord, our_maxvals, vis_output_dir, center, scale)

        logger.info('########################################')
        logger.info('{}'.format(self.cfg.EXPERIMENT_NAME))
        model_name = self.cfg.MODEL.NAME

        acc_printer = PredsAccPrinter(self.cfg, all_boxes, dataset, filenames, filenames_map,
                                      imgnums, model_name,
                                      result_output_dir, self._print_name_value)
        logger.info("====> Predicting key frame heatmaps by the backbone network")
        acc_printer(all_bb)
        logger.info("====> Predicting key frame heatmaps by the local warped hm")
        acc_printer(all_preds)
        tb_writer_dict["global_steps"] = self.global_steps

    def vis_info(self, logger, phase, epoch):
        if phase == TEST_PHASE:
            prefix_dir = "test"
        elif phase == TRAIN_PHASE:
            prefix_dir = "train"
        elif phase == VAL_PHASE:
            prefix_dir = "validate"
        else:
            prefix_dir = "inference"

        if isinstance(epoch, int):
            epoch = "model_{}".format(str(epoch))

        output_dir_base = osp.join(self.output_dir, epoch, prefix_dir,
                                   "use_gt_box" if self.cfg.VAL.USE_GT_BBOX else "use_precomputed_box")
        vis_output_dir = osp.join(output_dir_base, "vis")
        result_output_dir = osp.join(output_dir_base, "prediction_result")
        create_folder(vis_output_dir)
        create_folder(result_output_dir)
        logger.info("=> Vis Output Dir : {}".format(vis_output_dir))
        logger.info("=> Result Output Dir : {}".format(result_output_dir))

        if phase == VAL_PHASE:
            tensorboard_log_dir = osp.join(self.output_dir, epoch, prefix_dir, "tensorboard")
            self.tb_writer = SummaryWriter(log_dir=tensorboard_log_dir)

        if self.cfg.DEBUG.VIS_SKELETON:
            logger.info("=> VIS_SKELETON")
        if self.cfg.DEBUG.VIS_BBOX:
            logger.info("=> VIS_BBOX")
        return result_output_dir, vis_output_dir

    def vis_hook(self, image, preds_joints, preds_confidence, vis_output_dir, center, scale):
        cfg = self.cfg

        # prepare data
        coords = np.concatenate([preds_joints, preds_confidence], axis=-1)
        bboxes = []
        for index in range(len(center)):
            xyxy_bbox = cs2box(center[index], scale[index], pattern="xyxy")
            bboxes.append(xyxy_bbox)

        if cfg.DEBUG.VIS_SKELETON or cfg.DEBUG.VIS_BBOX:
            from engine.core.utils.vis_helper import draw_skeleton_in_origin_image
            draw_skeleton_in_origin_image(image, coords, bboxes, vis_output_dir,
                                          vis_skeleton=cfg.DEBUG.VIS_SKELETON,
                                          vis_bbox=cfg.DEBUG.VIS_BBOX,
                                          sure_threshold=0.1)


class PredsAccPrinter(object):
    def __init__(self, cfg, all_boxes, dataset, filenames, filenames_map, imgnums, model_name,
                 result_output_dir,
                 print_name_value_func):
        self.cfg = cfg
        self.all_boxes = all_boxes
        self.dataset = dataset
        self.filenames = filenames
        self.filenames_map = filenames_map
        self.imgnums = imgnums
        self.model_name = model_name
        self.result_output_dir = result_output_dir
        self.print_name_value_func = print_name_value_func

    def __call__(self, pred_result):
        name_values, perf_indicator = self.dataset.evaluate(self.cfg, pred_result,
                                                            self.result_output_dir,
                                                            self.all_boxes, self.filenames_map,
                                                            self.filenames, self.imgnums)
        if isinstance(name_values, list):
            for name_value in name_values:
                self.print_name_value_func(name_value, self.model_name)
        else:
            self.print_name_value_func(name_values, self.model_name)

        return name_values
