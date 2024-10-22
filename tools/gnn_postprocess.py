from yolox.utils import meshgrid
import os
import torch
import torchvision
from torch_geometric.nn import MessagePassing
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch



class EdgePropagation(MessagePassing):
            def __init__(self):
                super(EdgePropagation, self).__init__(aggr='add')  # 使用加法聚合信息

            def forward(self, x, edge_index, edge_weight):
                return self.propagate(edge_index, x=x, edge_weight=edge_weight)

            def message(self, x_j, edge_weight):
                return x_j * edge_weight

            def update(self, aggr_out):
                return aggr_out


class GNN_Postprocess:
    def __init__(self, img_size):
        self.color_structure = None
        self.boxinfo_structure = None
        self.topcls_calculate = None
        self.threshold = None

        self.img_h, self.img_w = img_size
        self.h_dynamic, self.w_dynamic = self.img_h//8, self.img_w//8
        self.get_distances()
        self.conf_model = EdgePropagation()

    def get_distances(self):
        coords = [[],[],[]]
        # 生成节点的三维坐标
        coords[0] = [(i, j, 0) for i in range(self.w_dynamic) for j in range(self.h_dynamic)]
        coords[1] = [(i, j, 0.1) if i % 2 == 0 and j % 2 == 0 else (float('nan'),float('nan'),float('nan'))
                        for i in range(self.w_dynamic) for j in range(self.h_dynamic)]
        coords[2] = [(i, j, 0.2) if i % 4 == 0 and j % 4 == 0 else (float('nan'),float('nan'),float('nan'))
                        for i in range(self.w_dynamic) for j in range(self.h_dynamic)]
        coords = torch.tensor(coords, dtype=torch.float)
        coords = coords.view(-1,3)
        distances = squareform(pdist(coords))
        # wxl：发现对角线上的点距离自动为0，无论对角线上的点坐标值为数值还是nan，这样不利于剔除无用距离，故将对角线上的点为nan处修改为nan。
        count = 0
        for i in range(3*int(self.w_dynamic*self.h_dynamic)):
            if i // int(self.w_dynamic*self.h_dynamic) == 1 \
            and ( 
                (((i % int(self.w_dynamic*self.h_dynamic)) % self.w_dynamic % 2) != 0)
                or 
                (((i % int(self.w_dynamic*self.h_dynamic)) // self.w_dynamic % 2) != 0)
                ):
                count += 1
                distances[i,i] = float("nan")
            elif i // int(self.w_dynamic*self.h_dynamic) == 2 \
            and ( 
                (((i % int(self.w_dynamic*self.h_dynamic)) % self.w_dynamic % 4) != 0)
                or 
                (((i % int(self.w_dynamic*self.h_dynamic)) // self.w_dynamic % 4) != 0)
                ):
                count += 1
                distances[i,i] = float("nan")
        self.distances = distances[~np.isnan(distances)].reshape((int(self.w_dynamic*self.h_dynamic*(1 + 1/4 + 1/16)))
                                                                ,(int(self.w_dynamic*self.h_dynamic*(1 + 1/4 + 1/16))))

    def run(self, img, gt, detect_output, threshold):
        self.threshold = threshold
        self.img, self.gt, self.detect_output = img, gt[0], detect_output[0]
        self.num_gt = self.gt.shape[0]

        self.gen_fg_area()
        self.color_structure_cls_structure()
        self.edge_relationship_passing()
        return self.ourpostproces2()

    def get_conf_weight(self, layer, m, n):
        distance = self.distances[m,n]
        color_m = self.color_structure[layer, m]
        color_n = self.color_structure[layer, n]
        if color_m == 1 and color_n == 1:
            conf_weight =  0.001 / distance
        elif color_m == 2 and color_n == 2:
            conf_weight = 0.001 / distance
        elif color_m == 2 and color_n == 1:
            conf_weight =  0.001 / distance
        else:
            conf_weight = 0.001 / distance
        conf_weight = min(max(conf_weight, 0.0), 1.0)
        return conf_weight

    def ourget_geometry_constraint(
        self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts
    ):
        """
            输入: 真值框 和 多尺度栅格信息
            输出: 真值框周围的栅格框 和 几何约束关系
            实现：根据真值框的位置，找出周围给定半径内锚框，作为第一个输出,选取在8倍降采样栅格空间中的1/10为直径。
                根据锚框和真值框的位置关系，判断锚框是否在真值框的周围，作为第二个输出
        """
        expanded_strides_per_image = expanded_strides[0]
        x_centers_per_image = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)
        y_centers_per_image = ((y_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)

        center_radius_w = self.img_w/8 *0.1 *0.5  #前景直径宽占最大栅格空间的宽的10%，半径就是再除以2
        center_radius_h = self.img_h/8 *0.1 *0.5
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

        self.fg_mask = is_in_centers.sum(dim=0) > 0
        self.geometry_relation = is_in_centers[:, self.fg_mask]


    def gen_fg_area(self):
        """
            输入:原来的demo经过原来的后处理的结果gt伪真值
            输出：前景锚框的位置
            实现:根据outputs1的位置,找出周围给定半径内锚框。
        """
        # outputs2 = outputs2[0]
        ourgt_bboxes_per_image = self.gt[:,:4]
        grids = []
        for k in [8, 16, 32]:
            self.detecthsize = int(self.img_h / k)
            self.detectwsize = int(self.img_w / k)
            grid = self.get_output_and_grid()
            grids.append(grid)
        expanded_stride0 = torch.tensor([8]).repeat(grids[0].shape[1]).cuda()
        expanded_stride1 = torch.tensor([16]).repeat(grids[1].shape[1]).cuda()
        expanded_stride2 = torch.tensor([32]).repeat(grids[2].shape[1]).cuda()
        expanded_strides = torch.cat([expanded_stride0, expanded_stride1, expanded_stride2], dim=0).unsqueeze(0)
        grids = torch.cat(grids, dim=1).cuda()
        x_shifts = grids[..., 0]
        y_shifts = grids[..., 1]

        self.ourget_geometry_constraint(
            ourgt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
        )
    

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
    

    #这个函数实现，根据cost（交幷比损失）选出在fg_area中top百分比的红框，剩下是黑框,,现在是设置为fg_area的30%
    def ourcolor_matching(self, cost, geometry_relation):
        num_gt = cost.size(0)
        matching_matrix = torch.zeros_like(cost, dtype=torch.bool)
        n_candidate_k = 0.3 * geometry_relation.sum(-1)  ###在不同的gt层，找出不同数量的红色框,现在是设置为fg_area的30%
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=int(n_candidate_k[gt_idx].item()), largest=False
            )
            matching_matrix[gt_idx][pos_idx] = True   ###在fg_map上标记了红框
        return matching_matrix 


    def get_output_and_grid(self):
        yv, xv = meshgrid([torch.arange(self.detecthsize), torch.arange(self.detectwsize)])
        grid = torch.stack((xv, yv), 2).view(1, 1, self.detecthsize, self.detectwsize, 2).to(torch.int64)
        grid = grid.view(1, -1, 2)
        return  grid
    
    #根据各种限制关系，得到了红框，黑框在num_gt个多尺度图上的排列关系和每个框的box信息。
    def color_structure_cls_structure(self):
        ourgt_bboxes_per_image = self.gt[:,0:4]
        ourpreds_bboxes_per_image = self.detect_output[self.fg_mask][:,0:4]
        ourpair_wise_ious = self.bboxes_iou(ourgt_bboxes_per_image, ourpreds_bboxes_per_image, False)
        pair_wise_ious_loss = - torch.log(ourpair_wise_ious + 1e-8)
        ourcost = (
            3.0 * pair_wise_ious_loss
            + float(1e6) * (~self.geometry_relation)
        )
        ###cost的计算交幷比不太够。
        #或许可以找新的交幷比
        matching_matrix = self.ourcolor_matching(ourcost, self.geometry_relation)
        # num_gt = matching_matrix.shape[0]  #这里的num_gt是根据原本的output伪真值的数量得来

        try:
            assert self.fg_mask.shape[0] == int(self.h_dynamic * self.w_dynamic * (1 + 1/4 + 1/16))
        except AssertionError:
            # 处理断言失败的情况
            print("box图构建有问题")
            
        #先构建一个num_gt*8400的图，用来放置红框，黑框，绿框颜色信息，后面转为num_gt*80*80的图
        self.color_structure = torch.zeros((self.num_gt, self.fg_mask.shape[0]),dtype = torch.float32)  
        for gt_layer in range(self.num_gt):
            tureboxcount = 0
            for i,value in enumerate(self.fg_mask):
                if value == True:  #在fg_aere中true的框，才有可能在matching_matrix中true
                    if matching_matrix[gt_layer, tureboxcount] == True:
                        self.color_structure[gt_layer,i] = 1  #在多尺度拼接图8400标记了红框：在fg_mask为true，在matching_matrix中为true
                    elif self.geometry_relation[gt_layer, tureboxcount] == True:
                        self.color_structure[gt_layer,i] = 2  #标记了黑框：在fg_mask为true，在geometry_relation中为true但是在matching_matrix中为false
                    tureboxcount += 1

                 
        #先构建一个num_gt*k_cls*2的图，用来放置前k_cls个类别的conf均值和类别idx
        k_cls =3  ##这个3是指目前选取所有类别中conf均值最大的3类
        self.topcls_calculate = torch.zeros((self.num_gt,k_cls, 2 ),dtype = torch.float)
        for gt_layer in range(self.num_gt):
            tobemeanidx = []
            for i in range(self.color_structure.shape[1]):
                if self.color_structure[gt_layer,i] == 1 or self.color_structure[gt_layer,i] == 2:
                    tobemeanidx.append(i)        
            allcls = self.detect_output[tobemeanidx,5:]
            mean_values = torch.mean(allcls, axis=0)##得到每一个cls的conf的平均值
            topcls, topidx= torch.topk(mean_values,k_cls)
            self.topcls_calculate[gt_layer] = torch.cat((topcls.unsqueeze(-1),topidx.unsqueeze(-1)),dim=-1)

     
    def get_edge_relationship(self, threshold_percentile=50):
        edge_indexes = []
        edge_weights = []
        for gt_layer in range(self.num_gt):
            node_colors = self.color_structure[gt_layer,:].cpu().numpy()
            red_indices = [i for i in range(node_colors.shape[0]) if node_colors[i] == 1]
            black_indices = [i for i in range(node_colors.shape[0]) if node_colors[i] == 2]

            if len(red_indices) > 1:
                red_threshold = np.percentile(self.distances[np.ix_(red_indices, red_indices)], threshold_percentile)
            else:
                red_threshold = float('inf')  # 避免无边的情况
            if len(black_indices) > 1:
                black_threshold = np.percentile(self.distances[np.ix_(black_indices, black_indices)], threshold_percentile)
            else:
                black_threshold = float('inf')

            edge_index = []
            conf_edge_weight = []
            # 红色节点之间的边
            for m in red_indices:
                for n in red_indices:
                    if m != n and self.distances[m, n] < red_threshold:
                        edge_index.append([m, n])
                        conf_weight  = self.get_conf_weight(gt_layer, m, n)
                        conf_edge_weight.append(conf_weight)

            # 黑色节点之间的边
            for m in black_indices:
                for n in black_indices:
                    if m != n and self.distances[m, n] < black_threshold:
                        edge_index.append([m, n])
                        conf_weight  = self.get_conf_weight(gt_layer, m, n)
                        conf_edge_weight.append(conf_weight)
                    
            # 红色给黑色传递，只有直接相连的边进行传递
            for m in red_indices:
                for n in black_indices:
                    if self.distances[m, n] == 1.0:  # 只传递直接相邻的边
                        edge_index.append([m, n])
                        conf_weight  = self.get_conf_weight(gt_layer, m, n)
                        conf_edge_weight.append(conf_weight)

            for m in black_indices:
                for n in red_indices:
                    if self.distances[m, n] == 1.0:  # 只传递直接相邻的边
                        edge_index.append([m, n])
                        conf_weight  = self.get_conf_weight(gt_layer, m, n)
                        conf_edge_weight.append(conf_weight)

            edge_index = torch.tensor(edge_index).t().contiguous()
            edge_weight = torch.tensor(conf_edge_weight, dtype=torch.float).unsqueeze(1) #增加维度，适应传递框架的要求
            edge_indexes.append(edge_index)
            edge_weights.append(edge_weight)

        return edge_indexes, edge_weights

    # 根据图结构进行传递参数，并且调用可视化函数，保存可视化图片
    def edge_relationship_passing(self):

        edge_indexes, edge_weights = self.get_edge_relationship()  #obj_conf的边关系传递图
        for gt_layer in range(self.num_gt):
            clses = self.topcls_calculate[gt_layer]
            edge_index = edge_indexes[gt_layer].to("cuda")
            edge_weight = edge_weights[gt_layer].to("cuda")
            for cls_index in range(clses.shape[0]):
                cls = int(clses[cls_index,1])
                data = self.detect_output[:,5+cls:6+cls]
                delta_data = self.conf_model(data, edge_index, edge_weight)
                self.detect_output[:,5+cls:6+cls] += delta_data


    def edge_relationship_visualize(self, coords, node_colors, edge_index, gt_layer, channel, save_path='./graph_visualization/'):
        colors = {1: 'r', 2: 'k'}
        plt.figure(figsize=(6, 6))
        # 创建保存目录
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 绘制节点
        for i, (x, y) in enumerate(coords):
            color_value = node_colors[int(x), int(y)]
            if color_value in colors:
                # plt.scatter(x, y, color=colors[color_value], s=200)
                # plt.text(x, y, str(i), fontsize=12, ha='center', va='center', color='white')
                plt.scatter( y,x, color=colors[color_value], s=200)
                plt.text(y,x, str(i), fontsize=12, ha='center', va='center', color='white')

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

    #将经过传递的信息，进行后半部分的后处理（置信度阈值筛选和nms处理），在这之前应该将三尺度叠加态重新变为8400的长度
    def ourpostproces2(self, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
        if self.threshold:
            conf_thre = self.threshold

        class_conf, class_id = torch.max(self.detect_output[:, 5:], 1, keepdim=True)
        conf_mask = (class_conf.squeeze()* self.detect_output[:, 4]>= conf_thre).squeeze()
        detections = torch.cat((self.detect_output[:, :4], class_conf, class_id), 1)
        detections = detections[conf_mask]

        nms_out_index = torchvision.ops.batched_nms(  #这种按照类别去计算交并比
            detections[:, :4],
            detections[:, 4],
            detections[:, 5],  #类别的index
            nms_thre,
        )

        detections = detections[nms_out_index]
        return detections

     

#将8400个obj_conf和class_conf的信息全部保留，只是处理了位置信息
def ourpostprocess1( prediction, num_classes, conf_thre=0.7):
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2 #cx,cy,w,h转为x1,y1,x2,y2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]
        return prediction
    
    


