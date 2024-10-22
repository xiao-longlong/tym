from torch_geometric.nn import MessagePassing
import torch

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


model = EdgePropagation()


x = torch.tensor([[1], [2], [3], [4], [5], [6]], dtype=torch.float32)

edge_index = torch.tensor([[0, 1],
                            [1, 2]], dtype=torch.int64)

edge_weight = torch.tensor([[2],[10]], dtype=torch.float32)

output = model(x, edge_index, edge_weight)

a = output

