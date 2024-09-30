# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image

# # 假设 color_graph_boxinfo 已经定义并具有指定的 shape
# color_graph_boxinfo = torch.randint(-1, 2, (2, 3, 80, 80, 8))  # 生成一个模拟数据

# # 提取 color_graph_boxinfo[0, :, :, :, 0]
# data = color_graph_boxinfo[0, :, :, :, 0].numpy()  # 将其转为 numpy 数组
# color_map = {
#     0: [0, 0, 0],        # 黑色
#     1: [255, 0, 0],      # 红色
#     -1: [105, 105, 105]  # 暗灰色
# }

# # 遍历每一层并可视化
# for i in range(data.shape[0]):
#     layer = data[i]
    
#     # 创建一个空白的 RGB 图像
#     img = np.zeros((layer.shape[0], layer.shape[1], 3), dtype=np.uint8)
    
#     # 根据值填充颜色
#     for key, color in color_map.items():
#         img[layer == key] = color
    
#     # 使用 PIL 保存图像
#     img_pil = Image.fromarray(img)
#     img_pil.save(f"layer_{i}.png")
    
#     # # 显示图像
#     # plt.imshow(img)
#     # plt.title(f"Layer {i}")
#     # plt.axis('off')  # 不显示坐标轴
#     # plt.show()



#--------------------------------------
#查看图片像素
from PIL import Image

# 打开图片
image = Image.open('/workspace/layer_2.png')

# 获取图片的宽度和高度
width, height = image.size
print(f"Image size: {width}x{height}")

# # 查看图片的某个像素的 RGB 值，例如 (x=10, y=10) 的像素
# pixel_value = image.getpixel((10, 10))
# print(f"Pixel at (10, 10): {pixel_value}")

# # 如果你想遍历整个图像并查看所有像素值
count = 0
for y in range(height):

    for x in range(width):
        pixel_value = image.getpixel((x, y))
        if pixel_value == (0, 0, 0) or pixel_value == (255, 0, 0):
            count += 1
            print(f"Pixel at ({x}, {y}): {pixel_value}")
print("红黑框个数",count)
