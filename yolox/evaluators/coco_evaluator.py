#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import contextlib
import io
import itertools
import json
import tempfile
import time
from collections import ChainMap, defaultdict
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm

import numpy as np

import torch

from yolox.utils.boxes import bboxes_iou
from yolox.utils import meshgrid
import torch.nn.functional as F


from yolox.data.datasets import COCO_CLASSES
# from yolox.data.datasets import COCO_CLASSES, GTSRB_CLASSES, VisDrone_CLASSES, SkyFusion_CLASSES
from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)




def per_class_AR_table(coco_eval, class_names=COCO_CLASSES, headers=["class", "AR"], colums=6):
    per_class_AR = {}
    recalls = coco_eval.eval["recall"]
    # dimension of recalls: [TxKxAxM]
    # recall has dims (iou, cls, area range, max dets)
    assert len(class_names) == recalls.shape[1]

    for idx, name in enumerate(class_names):
        recall = recalls[:, idx, 0, -1]
        recall = recall[recall > -1]
        ar = np.mean(recall) if recall.size else float("nan")
        per_class_AR[name] = float(ar * 100)

    num_cols = min(colums, len(per_class_AR) * len(headers))
    result_pair = [x for pair in per_class_AR.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table


def per_class_AP_table(coco_eval, class_names=COCO_CLASSES, headers=["class", "AP"], colums=6):
    per_class_AP = {}
    precisions = coco_eval.eval["precision"]
    # dimension of precisions: [TxRxKxAxM]
    # precision has dims (iou, recall, cls, area range, max dets)
    assert len(class_names) == precisions.shape[2]

    for idx, name in enumerate(class_names):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        per_class_AP[name] = float(ap * 100)

    num_cols = min(colums, len(per_class_AP) * len(headers))
    result_pair = [x for pair in per_class_AP.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table


class COCOEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self,
        dataloader,
        img_size: int,
        confthre: float,
        nmsthre: float,
        num_classes: int,
        testdev: bool = False,
        per_class_AP: bool = True,
        per_class_AR: bool = True,
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size: image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre: confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre: IoU threshold of non-max supression ranging from 0 to 1.
            per_class_AP: Show per class AP during evalution or not. Default to True.
            per_class_AR: Show per class AR during evalution or not. Default to True.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.testdev = testdev
        self.per_class_AP = per_class_AP
        self.per_class_AR = per_class_AR

    def ourpostprocess(self,prediction, num_classes):
        box_corner = prediction.new(prediction.shape)
        box_corner[:, 0] = prediction[:, 0] - prediction[:, 2] / 2
        box_corner[:, 1] = prediction[:, 1] - prediction[:, 3] / 2
        box_corner[:, 2] = prediction[:, 0] + prediction[:, 2] / 2
        box_corner[:, 3] = prediction[:, 1] + prediction[:, 3] / 2
        prediction[:, :4] = box_corner[:, :4]


        return prediction

    def get_output_and_grid(self, hsize, wsize):
        # grid = self.grids[k]

        # batch_size = output.shape[0]
        # n_ch = 5 + self.num_classes
        # hsize, wsize = output.shape[-2:]
        # if grid.shape[2:4] != output.shape[2:4]:
        yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
        grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).to(torch.int64)
        # self.grids[k] = grid

        # output = output.view(batch_size, 1, n_ch, hsize, wsize)
        # output = output.permute(0, 1, 3, 4, 2).reshape(
        #     batch_size, hsize * wsize, -1
        # )
        grid = grid.view(1, -1, 2)
        # output[..., :2] = (output[..., :2] + grid) * stride
        # output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return  grid
    
    def get_geometry_constraint(
        self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts,
    ):
        """
        Calculate whether the center of an object is located in a fixed range of
        an anchor. This is used to avert inappropriate matching. It can also reduce
        the number of candidate anchors so that the GPU memory is saved.
        """
        expanded_strides_per_image = expanded_strides[0]
        x_centers_per_image = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)
        y_centers_per_image = ((y_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)

        # in fixed center
        center_radius = 1.5
        center_dist = expanded_strides_per_image.unsqueeze(0) * center_radius
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0:1]) - center_dist
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0:1]) + center_dist
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1:2]) - center_dist
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1:2]) + center_dist

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        anchor_filter = is_in_centers.sum(dim=0) > 0
        geometry_relation = is_in_centers[:, anchor_filter]

        return anchor_filter, geometry_relation

    
    def oursimota_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        n_candidate_k = min(10, pair_wise_ious.size(1))
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)#计算每个预测出的锚框匹配几个真值框
        # deal with the case that one anchor matches multiple ground-truths
        if anchor_matching_gt.max() > 1:
            multiple_match_mask = anchor_matching_gt > 1
            _, cost_argmin = torch.min(cost[:, multiple_match_mask], dim=0)
            matching_matrix[:, multiple_match_mask] *= 0
            matching_matrix[cost_argmin, multiple_match_mask] = 1
        fg_mask_inboxes = anchor_matching_gt > 0
        num_fg = fg_mask_inboxes.sum().item()

        gt_matching_anchor = matching_matrix.sum(1)#计算每个真值框匹配几个预测出的锚框
        if gt_matching_anchor.max() > 1:  #匹配多个的解决方式
            multiple_match_mask2 = gt_matching_anchor > 1
            _ , cost_argmin2= torch.min(cost[multiple_match_mask2], dim=1)
            matching_matrix[multiple_match_mask2] *= 0
            matching_matrix[multiple_match_mask2, cost_argmin2] = 1
        fg_mask_inboxes2 = gt_matching_anchor > 0
        num_fg2 = fg_mask_inboxes2.sum().item()
        # fg_mask[fg_mask.clone()] = fg_mask_inboxes

        # matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        # gt_matched_classes = gt_classes[matched_gt_inds]

        # pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
        #     fg_mask_inboxes
        # ]
        # return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
        return num_fg2, matching_matrix 





    def tcls_ratio(self, ouroutputs_list):
        allfenzi = 0
        allfenmu = 0
        for ouridx, ouroutput in enumerate(ouroutputs_list):
            ouroutput = self.ourpostprocess(ouroutput, self.num_classes)

            ourgt_bboxes_per_image = torch.from_numpy(self.dataloader.dataset.annotations[ouridx][0][:,:4]).cuda()
            hsize, wsize = self.dataloader.dataset.annotations[ouridx][1]
            grids = []
            for k in [8, 16, 32]:
                ourhsize = int(hsize / k)
                ourwsize = int(wsize / k)
                # grid = self.get_output_and_grid(hsize, wsize, 'torch.cuda.HalfTensor')
                grid = self.get_output_and_grid(ourhsize, ourwsize)

                grids.append(grid)
            expanded_stride0 = torch.tensor([8]).repeat(grids[0].shape[1]).cuda()
            expanded_stride1 = torch.tensor([16]).repeat(grids[1].shape[1]).cuda()
            expanded_stride2 = torch.tensor([32]).repeat(grids[2].shape[1]).cuda()
            expanded_strides = torch.cat([expanded_stride0, expanded_stride1, expanded_stride2], dim=0).unsqueeze(0)
            grids = torch.cat(grids, dim=1).cuda()
            x_shifts = grids[..., 0]
            y_shifts = grids[..., 1]

            fg_mask, geometry_relation = self.get_geometry_constraint(
                ourgt_bboxes_per_image,
                expanded_strides,
                x_shifts,
                y_shifts,
            )
            ouroutput = ouroutput[fg_mask]


            ourbboxes_preds_per_image = ouroutput[:,:4]
            ourpair_wise_ious = bboxes_iou(ourgt_bboxes_per_image, ourbboxes_preds_per_image, False)
            pair_wise_ious_loss = -torch.log(ourpair_wise_ious + 1e-8)


            num_gt = ourgt_bboxes_per_image.size(0)
            ourclses_preds_per_image = ouroutput[:,5:5 + self.num_classes]
            cls_preds_ = ourclses_preds_per_image.unsqueeze(0).repeat(num_gt, 1, 1)

            _, class_pred = torch.max(ouroutput[:, 5: 5 + self.num_classes], 1, keepdim=True)
            ourclsidxes_preds_per_image = class_pred
            num_dt = ourclsidxes_preds_per_image.size(0)


            gt_classes = torch.from_numpy(self.dataloader.dataset.annotations[ouridx][0][:,-1]).cuda()
            ourgt_cls_per_image = torch.zeros((num_gt, self.num_classes)).cuda()
            ourgt_clsesidx_per_image = gt_classes.unsqueeze(-1)
            for i in range(self.num_classes):
                for j,ourgt_clsidx_per_image in enumerate(ourgt_clsesidx_per_image):
                    if ourgt_clsidx_per_image == i:
                        ourgt_cls_per_image[j,i] = 1
            gt_cls_per_image = ourgt_cls_per_image.unsqueeze(1).repeat(1, num_dt, 1)
                

            pair_wise_cls_loss = F.binary_cross_entropy(
                    cls_preds_,
                    gt_cls_per_image,
                    reduction="none"
                ).sum(-1)


            ourcost = (
                pair_wise_cls_loss
                + 3.0 * pair_wise_ious_loss
                + float(1e6) * (~geometry_relation)
            )

        
            num_fg2, matching_matrix = self.oursimota_matching(ourcost, ourpair_wise_ious, gt_classes, num_gt, fg_mask)
            ourindices = torch.nonzero(matching_matrix == 1)
            ourpoints = [(index[0].item(), index[1].item()) for index in ourindices]
            fenzi = 0
            for point in ourpoints:
                if ourclsidxes_preds_per_image.squeeze(-1)[point[1]] == gt_classes[point[0]]:
                    fenzi += 1
            allfenzi += fenzi
            allfenmu += num_fg2
        if allfenmu != 0:
            ourtcls_ratio = allfenzi / allfenmu
        return ourtcls_ratio


        




    def evaluate(
        self, model, distributed=False, half=False, trt_file=None,
        decoder=None, test_size=None, return_outputs=False
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        output_data = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        ouroutputs_list = []

        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)

                # skip the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())
                ouroutputs_list.extend(outputs)
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start
                outputs = postprocess(
                # outputs = ourpostprocess(
                    outputs, self.num_classes, self.confthre, self.nmsthre
                )
                # ouroutputs_list.extend(outputs)


                #postprocess后的outputs是个batchsize长度的列表，每个元素是图片的预测信息，shape是num_boxes*7，7分别是0-3是坐标，4-6是置信度、类别置信度，cls_index
                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end
            data_list_elem, image_wise_data = self.convert_to_coco_format(
                outputs, info_imgs, ids, return_outputs=True)
            data_list.extend(data_list_elem)
            output_data.update(image_wise_data)
            

        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if distributed:
            # different process/device might have different speed,
            # to make sure the process will not be stucked, sync func is used here.
            synchronize()
            data_list = gather(data_list, dst=0)
            output_data = gather(output_data, dst=0)
            data_list = list(itertools.chain(*data_list))
            output_data = dict(ChainMap(*output_data))
            torch.distributed.reduce(statistics, dst=0)

        tcls_ratio_tym = self.tcls_ratio(ouroutputs_list) 
        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()

        if return_outputs:
            return eval_results, output_data, tcls_ratio_tym
        return eval_results, tcls_ratio_tym

    def convert_to_coco_format(self, outputs, info_imgs, ids, return_outputs=False):
        data_list = []
        image_wise_data = defaultdict(dict)
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]

            image_wise_data.update({
                int(img_id): {
                    "bboxes": [box.numpy().tolist() for box in bboxes],
                    "scores": [score.numpy().item() for score in scores],
                    "categories": [
                        self.dataloader.dataset.class_ids[int(cls[ind])]
                        for ind in range(bboxes.shape[0])
                    ],
                }
            })

            bboxes = xyxy2xywh(bboxes)

            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)

        if return_outputs:
            return data_list, image_wise_data
        return data_list

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "NMS", "inference"],
                    [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            if self.testdev:
                json.dump(data_dict, open("./yolox_testdev_2017.json", "w"))
                cocoDt = cocoGt.loadRes("./yolox_testdev_2017.json")
            else:
                _, tmp = tempfile.mkstemp()
                json.dump(data_dict, open(tmp, "w"))
                cocoDt = cocoGt.loadRes(tmp)
            try:
                from yolox.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools.cocoeval import COCOeval

                logger.warning("Use standard COCOeval.")

            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            cat_ids = list(cocoGt.cats.keys())
            cat_names = [cocoGt.cats[catId]['name'] for catId in sorted(cat_ids)]
            if self.per_class_AP:
                AP_table = per_class_AP_table(cocoEval, class_names=cat_names)
                info += "per class AP:\n" + AP_table + "\n"
            if self.per_class_AR:
                AR_table = per_class_AR_table(cocoEval, class_names=cat_names)
                info += "per class AR:\n" + AR_table + "\n"
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info
