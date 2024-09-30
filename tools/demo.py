#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
from yolox.utils import meshgrid

import argparse
import os
import time
from loguru import logger

import cv2

import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES, GTSRB_CLASSES, VisDrone_CLASSES, SkyFusion_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis


from torch_geometric.nn import MessagePassing
import numpy as np
import random
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import os

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment_name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./assets/dog.jpg", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--datasets_classes",
        dest="datasets_classes",
        default="COCO",
        type=str,
        help="datasets classes to use.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser


class EdgePropagation(MessagePassing):
            def __init__(self):
                super(EdgePropagation, self).__init__(aggr='add')  # 使用加法聚合信息

            def forward(self, x, edge_index, edge_weight):
                # 进行消息传递
                return self.propagate(edge_index, x=x, edge_weight=edge_weight)

            def message(self, x_j, edge_weight):
                # 消息传递，源节点的特征 * 边的权重
                return x_j * edge_weight

            def update(self, aggr_out):
                # 聚合后的输出直接返回
                return aggr_out



def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        # if trt_file is not None:
        #     from torch2trt import TRTModule

        #     model_trt = TRTModule()
        #     model_trt.load_state_dict(torch.load(trt_file))

        #     x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
        #     self.model(x)
        #     self.model = model_trt
    
    #################################################################################
    # 可视化图节点的分布及颜色并绘制箭头
    def visualize_graph(self, coords, node_colors, edge_index, gt_layer, channel, save_path='./graph_visualization/'):
        colors = {1: 'r', 2: 'k'}
        plt.figure(figsize=(6, 6))

        # 创建保存目录
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 绘制节点
        for i, (x, y) in enumerate(coords):
            color_value = node_colors[int(x), int(y)]
            if color_value in colors:
                plt.scatter(x, y, color=colors[color_value], s=200)
                plt.text(x, y, str(i), fontsize=12, ha='center', va='center', color='white')

        # 绘制带箭头的边，根据节点颜色决定箭头颜色
        for (i, j) in edge_index.t().numpy():
            start, end = coords[i], coords[j]
            
            if node_colors[int(start[0]), int(start[1])] == 1 and node_colors[int(end[0]), int(end[1])] == 1:
                arrow_color = 'red'  # 红色节点之间用红色箭头
            elif node_colors[int(start[0]), int(start[1])] == 2 and node_colors[int(end[0]), int(end[1])] == 2:
                arrow_color = 'black'  # 黑色节点之间用黑色箭头
            else:
                arrow_color = 'blue'  # 红色与黑色节点之间用蓝色箭头
            
            arrow = FancyArrowPatch(start, end, color=arrow_color, arrowstyle='->', mutation_scale=20, lw=0.5)
            plt.gca().add_patch(arrow)

        plt.grid(True)

        # 保存图像
        save_file = f'{save_path}gt_{gt_layer}_channel_{channel}.png'
        plt.savefig(save_file)
        print(f"Graph with arrows saved to {save_file}")

    

    def create_graph_from_tensor(self, tensor, num_gt, h_dynamic, w_dynamic, threshold_percentile=50):
            graphs = []

            for gt_layer in range(num_gt):
                for channel in range(3):
                    node_colors = tensor[gt_layer, channel, :, :, 0].cpu().numpy()

                    # 生成节点的二维坐标
                    coords = [(i, j) for i in range(h_dynamic) for j in range(w_dynamic)]
                    coords = torch.tensor(coords, dtype=torch.float)

                    # 将颜色1表示为红色节点，2表示为黑色节点
                    red_indices = [(i, j) for i in range(h_dynamic) for j in range(w_dynamic) if node_colors[i, j] == 1]
                    black_indices = [(i, j) for i in range(h_dynamic) for j in range(w_dynamic) if node_colors[i, j] == 2]

                    red_indices_flat = [i * w_dynamic + j for i, j in red_indices]
                    black_indices_flat = [i * w_dynamic + j for i, j in black_indices]

                    # 计算节点之间的欧氏距离
                    distances = squareform(pdist(coords))

                    # 设置红色和黑色节点的边关系阈值
                    if len(red_indices_flat) > 1:
                        red_threshold = np.percentile(distances[np.ix_(red_indices_flat, red_indices_flat)], threshold_percentile)
                    else:
                        red_threshold = float('inf')  # 避免无边的情况

                    if len(black_indices_flat) > 1:
                        black_threshold = np.percentile(distances[np.ix_(black_indices_flat, black_indices_flat)], threshold_percentile)
                    else:
                        black_threshold = float('inf')

                    edge_index = []
                    edge_weight = []

                    # 红色节点之间的边
                    for i in red_indices_flat:
                        for j in red_indices_flat:
                            if i != j and distances[i, j] < red_threshold:
                                edge_index.append([i, j])
                                edge_weight.append(random.random())  # 随机分配边的权重

                    # 黑色节点之间的边
                    for i in black_indices_flat:
                        for j in black_indices_flat:
                            if i != j and distances[i, j] < black_threshold:
                                edge_index.append([i, j])
                                edge_weight.append(random.random())  # 随机分配边的权重

                    # 红色和黑色节点之间只有直接相连的边
                    for i in red_indices_flat:
                        for j in black_indices_flat:
                            if distances[i, j] == 1.0:  # 只传递直接相邻的边
                                edge_index.append([i, j])
                                edge_weight.append(random.random())  # 随机分配边的权重

                    edge_index = torch.tensor(edge_index).t().contiguous()
                    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

                    # 每个节点初始化一个特征值
                    x = torch.rand((h_dynamic * w_dynamic, 1))

                    # 保存该通道的图结构信息
                    graphs.append((x, edge_index, edge_weight, node_colors, coords))

            return graphs
    
    def shiyixia(self, tensor):
            # 创建图
            num_gt,h_dynamic, w_dynamic = tensor.size(0), tensor.size(2), tensor.size(3)
            graphs = self.create_graph_from_tensor(tensor, num_gt, h_dynamic, w_dynamic)

            # 创建 GNN 模型
            model = EdgePropagation()

            # 遍历每个图
            for gt_layer in range(num_gt):
                for channel in range(3):
                    x, edge_index, edge_weight, node_colors, coords = graphs[gt_layer * 3 + channel]

                    # 进行消息传递
                    output = model(x, edge_index, edge_weight)

                    # 可视化每个图
                    self.visualize_graph(coords, node_colors, edge_index, gt_layer, channel)
    
    def color_graph_boxinfo(self,img, outputs2, fg_mask, matching_matrix, geometry_relation):
        num_gt = matching_matrix.shape[0]
        hsize, wsize = img.size()[2:4]
        baseh = int(hsize/8)
        basew = int(wsize/8)
        try:
            assert fg_mask.shape[0] == int(baseh * basew * (1 + 1/4 + 1/16))
        except AssertionError:
            # 处理断言失败的情况
            print("box图构建有问题")
        ##构建3*80*80*7的box图，放置了8400个锚框的位置类别置信度信息
        box_graph_stru = torch.zeros((3,baseh,basew,outputs2.shape[1]),dtype = torch.float32)
        for idx in range(fg_mask.shape[0]):
            if idx < baseh*basew: #0-6399
                rows = int(idx // basew) #行数
                cols = int(idx % basew)
                box_graph_stru[0,rows,cols,:] =  outputs2[idx,:]
            elif idx < int(baseh * basew * (1 + 1/4 )):  #6400-7999
                # rows = 2*(int((2 * (idx-baseh*basew))// basew))
                # cols = int((2 * (idx-baseh*basew)) % basew)
                rows = 2 * (int((idx-baseh*basew) // int(basew/2)))
                cols = 2 * (int((idx-baseh*basew) % int(basew/2)))
                box_graph_stru[1,rows,cols,:] =  outputs2[idx,:]
            else:     #8000-8399
                # rows = 4*(int((4 * (idx-baseh*basew*(1 + 1/4 )) )// basew))
                # cols = int((4 * (idx-baseh*basew*(1 + 1/4 )) )% basew)
                rows = 4 * (int((idx-baseh*basew*(1 + 1/4 )) // int(basew/4)))
                cols = 4 * (int((idx-baseh*basew*(1 + 1/4 )) % int(basew/4)))
                box_graph_stru[2,rows,cols,:] =  outputs2[idx,:]  
        
        
        #先构建一个num_gt*8400的图，用来放置红框，黑框，绿框，后面再转换成num_gt*3*80*80的图   
        color_mark = torch.zeros((num_gt, fg_mask.shape[0]),dtype = torch.float32)  
        for gt_layer in range(num_gt):
            tureboxcount = 0
            for i,value in enumerate(fg_mask):
                if value == True:  #在fg_mask中true的框，才有可能在matching_matrix中true
                    if matching_matrix[gt_layer, tureboxcount] == True:
                        color_mark[gt_layer,i] = 1  #在多尺度拼接图8400标记了红框
                    elif geometry_relation[gt_layer, tureboxcount] == True:
                        color_mark[gt_layer,i] = 2  #标记了黑框，在geometry_relation中为true但是在matching_matrix中为false
                    else:
                        color_mark[gt_layer,i] = 0   #标记了暗框
                    tureboxcount += 1
                else:
                    color_mark[gt_layer,i] = 0  #标记了暗框（不在matching_matrix范围内的框,暗灰色    
                
                
        #转换成num_gt*3*80*80的图，并且把且只红框和黑框对应的位置类别置信度信息放进去
        color_graph_boxinfo = torch.zeros((num_gt, 3, baseh, basew, (1 + outputs2.shape[1]) ),dtype = torch.float32)
        for gt_layer in range(num_gt):  #遍历num_gt
            for idx in range(color_mark.shape[1]): 
                if idx < baseh*basew: #0-6399
                    rows = int(idx // basew) #行数
                    cols = int(idx % basew)
                    color_graph_boxinfo[gt_layer, 0, rows, cols, 0] =  color_mark[gt_layer,idx]
                    if color_mark[gt_layer,idx] != 0:
                        color_graph_boxinfo[gt_layer, 0, rows, cols, 1:] =  box_graph_stru[0, rows, cols, :]
                elif idx < int(baseh * basew * (1 + 1/4 )):  #6400-7999
                    # rows = 2*(int((2 * (idx-baseh*basew))// basew))
                    # cols = int((2 * (idx-baseh*basew)) % basew)
                    rows = 2 * (int((idx-baseh*basew) // int(basew/2)))
                    cols = 2 * (int((idx-baseh*basew) % int(basew/2)))
                    color_graph_boxinfo[gt_layer, 1, rows, cols, 0] =  color_mark[gt_layer,idx]
                    if color_mark[gt_layer,idx] != 0:
                        color_graph_boxinfo[gt_layer, 1, rows, cols, 1:] =  box_graph_stru[1, rows, cols, :]
                else:     #8000-8399
                    # rows = 4*(int((4 * (idx-baseh*basew*(1 + 1/4 )) )// basew))
                    # cols = int((4 * (idx-baseh*basew*(1 + 1/4 )) )% basew)
                    rows = 4 * (int((idx-baseh*basew*(1 + 1/4 )) // int(basew/4)))
                    cols = 4 * (int((idx-baseh*basew*(1 + 1/4 )) % int(basew/4)))
                    
                    color_graph_boxinfo[gt_layer, 2, rows, cols, 0] =  color_mark[gt_layer,idx]
                    if color_mark[gt_layer,idx] != 0:
                        color_graph_boxinfo[gt_layer, 2, rows, cols, 1:] =  box_graph_stru[2, rows, cols, :]
        #可视化
        import numpy as np
        import matplotlib.pyplot as plt
        from PIL import Image
        data = color_graph_boxinfo[0, :, :, :, 0].numpy()  # 将其转为 numpy 数组
        color_map = {
            2: [0, 0, 0],        # 黑色
            1: [255, 0, 0],      # 红色
            0: [0, 255, 0]     # 绿色
        }

        # 遍历每一层并可视化
        for i in range(data.shape[0]):
            layer = data[i]
            
            # 创建一个空白的 RGB 图像
            img = np.zeros((layer.shape[0], layer.shape[1], 3), dtype=np.uint8)
            
            # 根据值填充颜色
            for key, color in color_map.items():
                img[layer == key] = color
            
            # 使用 PIL 保存图像
            img_pil = Image.fromarray(img)
            img_pil.save(f"layer_{i}.png")
            
                        
        return color_graph_boxinfo
            
            
            

    def bboxes_iou(self, bboxes_a, bboxes_b, xyxy=True):
        if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
            raise IndexError

        if xyxy:
            tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
            br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
            area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
            area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
        else:
            tl = torch.max(
                (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
            )
            br = torch.min(
                (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
            )

            area_a = torch.prod(bboxes_a[:, 2:], 1)
            area_b = torch.prod(bboxes_b[:, 2:], 1)
        en = (tl < br).type(tl.type()).prod(dim=2)
        area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
        return area_i / (area_a[:, None] + area_b - area_i)
    


    #这个函数实现，根据交幷比的值，找出红框和黑框
    def ourcolor_matching(self, cost, geometry_relation):
        num_gt = cost.size(0)
        matching_matrix = torch.zeros_like(cost, dtype=torch.bool)
        n_candidate_k = 0.3 * geometry_relation.sum(-1)  ###在不同的gt层，找出不同数量的候选框
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=int(n_candidate_k[gt_idx].item()), largest=False
            )
            matching_matrix[gt_idx][pos_idx] = True   ###在fg_map上标记了红框

        return matching_matrix 


    def get_output_and_grid(self, hsize, wsize):
        yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
        grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).to(torch.int64)
        grid = grid.view(1, -1, 2)
        return  grid
    def ourget_geometry_constraint(
        self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, hsize, wsize
    ):
        expanded_strides_per_image = expanded_strides[0]
        x_centers_per_image = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)
        y_centers_per_image = ((y_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)

        # in fixed center
        # center_radius = 1.5
        center_radius_w = wsize/8 *0.1 *0.5  #前景直径宽占最大栅格空间的宽的10%，半径就是再除以2
        center_radius_h = hsize/8 *0.1 *0.5
        center_dist_w = expanded_strides_per_image.unsqueeze(0) * center_radius_w
        center_dist_h = expanded_strides_per_image.unsqueeze(0) * center_radius_h

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0:1]) - center_dist_w
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0:1]) + center_dist_w
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1:2]) - center_dist_h
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1:2]) + center_dist_h

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0

        anchor_filter = is_in_centers.sum(dim=0) > 0
        geometry_relation = is_in_centers[:, anchor_filter]

        return anchor_filter, geometry_relation

    #这个函数打算根据两个输入（一个是原来的demo经过原来的后处理的结果outputs1伪真值，一个是没经过confmask和nms的8400个框outputs2）,根据outputs1的位置，找出周围给定半径内锚框。
    def gen_fg_area(self,img,outputs1):
        outputs1 = outputs1[0]
        # outputs2 = outputs2[0]
        ourgt_bboxes_per_image = outputs1[:,:4]
        hsize, wsize = img.shape[2:4]
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

        fg_mask, geometry_relation = self.ourget_geometry_constraint(
            ourgt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            hsize,
            wsize
        )
        # outputs2 = outputs2[fg_mask]
        return fg_mask , geometry_relation
            ###返回的outputs2,geometry_relation是一张图片的数据
    

    #这个函数输入为特定半径内的锚框，根据他们距离outputs1框的距离，区分亮框（离目标近）和暗框（离目标远）
    def gen_color_structure(self, img, outputs1, fg_mask ,outputs2, geometry_relation):
        outputs1 = outputs1[0]
        outputs2 = outputs2[0]
        #fg_anchors, outputs1,geometry_relations都是一张图的数据，因为推理没有batchsize
        ourgt_bboxes_per_image = outputs1[:,0:4]
        ourbboxes_preds_per_image = outputs2[fg_mask][:,0:4]
        ourpair_wise_ious = self.bboxes_iou(ourgt_bboxes_per_image, ourbboxes_preds_per_image, False)
        pair_wise_ious_loss = - torch.log(ourpair_wise_ious + 1e-8)
        ourcost = (
            3.0 * pair_wise_ious_loss
            + float(1e6) * (~geometry_relation)
        )
        ###cost的计算交幷比不太够。
        #或许可以找新的交幷比

        matching_matrix = self.ourcolor_matching(ourcost, geometry_relation)
        color_graph_boxinfo = self.color_graph_boxinfo(img, outputs2, fg_mask, matching_matrix, geometry_relation)


        

        # 主程序
        

            # # 假设 tensor 的维度为 num_gt * 3 * h_dynamic * w_dynamic_8
        # num_gt = 1
        # h_dynamic, w_dynamic = 80, 80
            # tensor = torch.randint(1, 3, (num_gt, 3, h_dynamic, w_dynamic, 1))  # 模拟的 tensor

        self.shiyixia(color_graph_boxinfo)







        return color_graph_boxinfo






    #根据不同颜色的框，构建边关系。
    def gen_gnn_rela(color_graph_boxinfo):
        
        return 0


    def ourpostprocess(self, prediction, num_classes, conf_thre=0.7):
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):

            # If none are remaining => process next image
            if not image_pred.size(0):
                continue
            # Get score and class with highest confidence,class_conf是置信度，class_pred是类别的index
            class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
            if not detections.size(0):
                continue
     
            if output[i] is None:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))
  

        return output





    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            #outputs1是原本的outputs,经过了nms处理
            outputs1 = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )

            outputs2 = self.ourpostprocess(
                outputs, self.num_classes, self.confthre
                )

            fg_mask , geometry_relation = self.gen_fg_area(img, outputs1)  #得到fg范围
            color_graph_boxinfo = self.gen_color_structure(img, outputs1, fg_mask, outputs2, geometry_relation)
            

            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs1, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    if args.save_result:
        save_folder = os.path.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        os.makedirs(save_folder, exist_ok=True)
        if args.demo == "video":
            save_path = os.path.join(save_folder, os.path.basename(args.path))
        else:
            save_path = os.path.join(save_folder, "camera.mp4")
        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            if args.save_result:
                vid_writer.write(result_frame)
            else:
                cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
                cv2.imshow("yolox", result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        # exp.test_size = (args.tsize, args.tsize)
        # exp.tett_size = (1344,800)
        exp.tett_size = (960,544)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    if args.datasets_classes == "COCO":
        pass
    elif args.datasets_classes == "GTSRB":
        COCO_CLASSES = GTSRB_CLASSES
    elif args.datasets_classes == "VisDrone":
        COCO_CLASSES = VisDrone_CLASSES
    elif args.datasets_classes == "SkyFusion":
        COCO_CLASSES = SkyFusion_CLASSES

    predictor = Predictor(
        model, exp, COCO_CLASSES, trt_file, decoder,
        args.device, args.fp16, args.legacy,
    )
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, args.path, current_time, args.save_result)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)
