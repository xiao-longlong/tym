import torch
from torch_geometric.nn import MessagePassing

class EdgePropagation(MessagePassing):
            def __init__(self):
                super(EdgePropagation, self).__init__(aggr='add')  # 使用加法聚合信息

            def forward(self, x, edge_index, edge_weight):
                return self.propagate(edge_index, x=x, edge_weight=edge_weight)

            def message(self, x_j, edge_weight):
                return x_j * edge_weight

            def update(self, aggr_out):
                return aggr_out

device='cuda:0'

data = torch.tensor([[0.059131],
        [0.067866],
        [0.087000],
        [0.017214],
        [0.015460],
        [0.017203]], device=device)

edge_index = torch.tensor([[0, 2, 1, 0, 3, 1],
                            [2, 4, 0, 2, 1, 3]], device=device)
edge_weight = torch.tensor([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]], device=device)
model = EdgePropagation().cuda()
output = model(data, edge_index, edge_weight)
a = output