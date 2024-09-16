import os
from PIL import Image, ImageDraw

# 图像尺寸
image_width = 1360
image_height = 800

# 标注文件路径（指定图片的txt文件）
annotation_file = "/workspace/datasets/GTSRBYOLO/00001.txt"  # 替换为您想查看的标注文件

# 图片文件路径
image_file = "/workspace/datasets/GTSRBYOLO/00001.png"  # 替换为您想查看的图片

# 打开图片
img = Image.open(image_file)
draw = ImageDraw.Draw(img)

# 读取标注文件内容
with open(annotation_file, 'r') as file:
    for line in file:
        parts = line.strip().split()
        class_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:])

        # 计算出原图像上的x1, y1, x2, y2
        x1 = (cx - w / 2) * image_width
        y1 = (cy - h / 2) * image_height
        x2 = (cx + w / 2) * image_width
        y2 = (cy + h / 2) * image_height

        # 绘制矩形框到图像上，并标注类别信息
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), f"Class {class_id}", fill="red")

# 保存或显示图像
output_image_path = "/workspace/datasets/GTSRBYOLO/0.png"
img.save(output_image_path)
img.show()  # 直接显示图像（如果运行环境支持）
