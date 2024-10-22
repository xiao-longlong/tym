# import torch
# AA=torch.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]],[[13,14,15],[16,17,18]],[[19,20,21],[22,23,24]]])
# BB=torch.tensor([[[1],[2],[3]],[[4],[5],[6]],[[7],[8],[9]],[[10],[11],[12]]])
# result = torch.einsum('klm,kmn->kln', AA, BB)
# print("AA",AA.shape)
# print("BB",BB.shape)
# print("result.shape",result.shape)
# print("result",result)

import torch
import torch.nn as nn

# # 创建两个示例 Tensor
# # 第一个 Tensor 形状为 (1, 2, 3, 4)，元素取值在 30 以内且不重复
# tensor1 = torch.tensor([[[[1, 2, 3, 4],
#                           [5, 6, 7, 8],
#                           [9, 10, 11, 12]],

#                          [[13, 14, 15, 16],
#                           [17, 18, 19, 20],
#                           [21, 22, 23, 24]]]], dtype=torch.uint8)

# # 第二个 Tensor 形状为 (1, 1, 3, 4)，元素取值在 30 以内且不重复
# tensor2 = torch.tensor([[[[1, 2, 3, 4],
#                           [5, 6, 7, 8],
#                           [9, 10, 11, 12]]]], dtype=torch.uint8)

# # 逐元素相乘，自动广播
# result = tensor1 * tensor2
# 打印结果
# print("Tensor 1:")
# print(tensor1d.shape)
# print("Tensor 2:")
# print(tensor2)
# print("Result:")
# print(result)


class BaseConv1d(nn.Module):
    """A Conv1d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
    def forward(self, x):
        return self.conv(x)

    def fuseforward(self, x):
        return self.act(self.conv(x))
# tensor1d1 = torch.tensor([[[[1]],[[2]],[[3]],[[4]],[[5]]],[[[6]],[[7]],[[8]],[[9]],[[10]]]], dtype=torch.float32)
tensor1d2 = torch.tensor([[1,2,3,4,5],[6,7,8,9,10]], dtype=torch.float32)


co1d = BaseConv1d(
    in_channels=2,
    out_channels=1,
    ksize=1,
    stride=1,
)
bn1d = nn.BatchNorm1d(1)


result2 = co1d(tensor1d2)
result2 = bn1d(result2)

print("卷积核权重",co1d.conv.weight)
print("卷积核偏置",co1d.conv.bias)
print("Tensor 2:")
print(tensor1d2.shape)
print("Result 2:")
print(result2)
print(result2.shape)

