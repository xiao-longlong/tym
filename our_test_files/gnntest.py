# import torch
# import torch.nn.functional as F
# from torch_geometric.data import Data
# from torch_geometric.nn import MessagePassing
# from torch_geometric.utils import add_self_loops
# import numpy as np
# import random
# from scipy.spatial.distance import pdist, squareform

# # 定义图神经网络的边传递模块
# class EdgePropagation(MessagePassing):
#     def __init__(self):
#         super(EdgePropagation, self).__init__(aggr='add')  # 使用加法聚合信息

#     def forward(self, x, edge_index, edge_weight):
#         # 进行消息传递
#         return self.propagate(edge_index, x=x, edge_weight=edge_weight)

#     def message(self, x_j, edge_weight):
#         # 消息传递，源节点的特征 * 边的权重
#         return x_j * edge_weight

#     def update(self, aggr_out):
#         # 聚合后的输出直接返回
#         return aggr_out

# # 生成4x4网格中的16个节点，并随机设置6个红色节点
# def create_graph():
#     num_nodes = 16  # 16 个节点
#     grid_size = (4, 4)  # 4x4 网格

#     # 生成节点的二维坐标
#     coords = [(i, j) for i in range(grid_size[0]) for j in range(grid_size[1])]
#     coords = torch.tensor(coords, dtype=torch.float)

#     # 随机选出6个红色节点，其他为黑色节点
#     red_indices = random.sample(range(num_nodes), 6)
#     node_colors = ['red' if i in red_indices else 'black' for i in range(num_nodes)]

#     # 计算节点之间的欧氏距离
#     distances = squareform(pdist(coords))

#     # 设置红色和黑色节点的边关系
#     red_threshold = np.percentile(distances[np.ix_(red_indices, red_indices)], 50)  # 红色节点50%的距离阈值
#     black_indices = [i for i in range(num_nodes) if i not in red_indices]
#     black_threshold = np.percentile(distances[np.ix_(black_indices, black_indices)], 50)  # 黑色节点50%的距离阈值

#     # 构建边：红色节点和黑色节点分别根据阈值过滤
#     edge_index = []
#     edge_weight = []
    
#     # 红色节点之间的边
#     for i in red_indices:
#         for j in red_indices:
#             if i != j and distances[i, j] < red_threshold:
#                 edge_index.append([i, j])
#                 edge_weight.append(random.random())  # 随机分配边的权重

#     # 黑色节点之间的边
#     for i in black_indices:
#         for j in black_indices:
#             if i != j and distances[i, j] < black_threshold:
#                 edge_index.append([i, j])
#                 edge_weight.append(random.random())  # 随机分配边的权重

#     # 红色和黑色节点之间只有直接相连的边
#     for i in red_indices:
#         for j in black_indices:
#             if distances[i, j] == 1.0:  # 只传递直接相邻的边
#                 edge_index.append([i, j])
#                 edge_weight.append(random.random())  # 随机分配边的权重

#     edge_index = torch.tensor(edge_index).t().contiguous()
#     edge_weight = torch.tensor(edge_weight, dtype=torch.float)

#     # 每个节点初始化一个特征值
#     x = torch.rand((num_nodes, 1))

#     return x, edge_index, edge_weight, node_colors, coords

# # 可视化图节点的分布及颜色
# def visualize_graph(coords, node_colors, edge_index, save_path='graph_visualization_with_arrows.png'):
#     import matplotlib.pyplot as plt
#     from matplotlib.patches import FancyArrowPatch

#     colors = {'red': 'r', 'black': 'k'}
#     plt.figure(figsize=(6, 6))

#     # 绘制节点
#     for i, (x, y) in enumerate(coords):
#         plt.scatter(x, y, color=colors[node_colors[i]], s=200)
#         plt.text(x, y, str(i), fontsize=12, ha='center', va='center', color='white')

#     # 绘制带箭头的边，根据节点颜色决定箭头颜色
#     for (i, j) in edge_index.t().numpy():
#         start, end = coords[i], coords[j]
        
#         # 红色节点之间的边用红色箭头
#         if node_colors[i] == 'red' and node_colors[j] == 'red':
#             arrow_color = 'red'
#         # 黑色节点之间的边用黑色箭头
#         elif node_colors[i] == 'black' and node_colors[j] == 'black':
#             arrow_color = 'black'
#         # 红色和黑色节点之间的边用蓝色箭头
#         else:
#             arrow_color = 'blue'
        
#         # 使用FancyArrowPatch绘制箭头
#         arrow = FancyArrowPatch(start, end, color=arrow_color, arrowstyle='->', mutation_scale=40, lw=0.5)
#         plt.gca().add_patch(arrow)

#     plt.grid(True)

#     # 保存到本地文件
#     plt.savefig(save_path)
#     print(f"Graph with arrows saved to {save_path}")
#     plt.show()



# # 主程序
# def main():
#     # 创建图
#     x, edge_index, edge_weight, node_colors, coords = create_graph()

#     # 创建 GNN 模型
#     model = EdgePropagation()

#     # 进行消息传递
#     output = model(x, edge_index, edge_weight)

#     # # 输出结果
#     # print("输入节点特征：", x.squeeze())
#     # print("边权重：", edge_weight)
#     # print("输出节点特征：", output.squeeze())

#     # 可视化节点分布
#     visualize_graph(coords, node_colors, edge_index, save_path='./graph_with_arrows.png')

# if __name__ == "__main__":
#     main()



##########---------------------------------------------------------
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import numpy as np
import random
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import os

# 定义图神经网络的边传递模块
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

# 创建节点并根据传入的 tensor 数据生成边关系
def create_graph_from_tensor(tensor, num_gt, h_dynamic, w_dynamic, threshold_percentile=50):
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

# 可视化图节点的分布及颜色并绘制箭头
def visualize_graph(coords, node_colors, edge_index, gt_layer, channel, save_path='./graph_visualization/'):
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
    plt.show()

# 主程序
def main(tensor, num_gt, h_dynamic, w_dynamic):
    # 创建图
    graphs = create_graph_from_tensor(tensor, num_gt, h_dynamic, w_dynamic)

    # 创建 GNN 模型
    model = EdgePropagation()

    # 遍历每个图
    for gt_layer in range(num_gt):
        for channel in range(3):
            x, edge_index, edge_weight, node_colors, coords = graphs[gt_layer * 3 + channel]

            # 进行消息传递
            output = model(x, edge_index, edge_weight)

            # 可视化每个图
            visualize_graph(coords, node_colors, edge_index, gt_layer, channel)
    return output
if __name__ == "__main__":
    # 假设 tensor 的维度为 num_gt * 3 * h_dynamic * w_dynamic_8
    num_gt = 2
    h_dynamic, w_dynamic = 4, 4
    tensor = torch.randint(1, 3, (num_gt, 3, h_dynamic, w_dynamic, 1))  # 模拟的 tensor

    main(tensor, num_gt, h_dynamic, w_dynamic)
