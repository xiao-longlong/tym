import os
from PIL import Image

# 读取标注文件
input_annotation_file = "/workspace/datasets/FullIJCNN2013/gt.txt"  # 假设标注信息保存在 annotations.txt 文件中

# 图像尺寸
image_width = 1360
image_height = 800

# 创建存储转换后的YOLO格式标注的目录
output_dir = "/workspace/datasets/GTSRBYOLO"


os.makedirs(output_dir, exist_ok=True)

# 用于存储每个文件的标注数据
file_annotations = {}

# 读取文件内容
with open(input_annotation_file, 'r') as file:
    for line in file:
        parts = line.strip().split(';')
        filename = parts[0].replace('.ppm', '.txt')  # 转换成 .txt 文件
        x1, y1, x2, y2, class_id = map(int, parts[1:])

        # 计算中心点和宽高
        cx = (x1 + x2) / 2 / image_width
        cy = (y1 + y2) / 2 / image_height
        w = (x2 - x1) / image_width
        h = (y2 - y1) / image_height

        # 将值保留6位小数
        cx = f"{cx:.6f}"
        cy = f"{cy:.6f}"
        w = f"{w:.6f}"
        h = f"{h:.6f}"

        # 生成YOLO格式的内容
        yolo_format = f"{class_id} {cx} {cy} {w} {h}\n"

        # 将标注数据存储在字典中，按文件名分类
        if filename not in file_annotations:
            file_annotations[filename] = []
        file_annotations[filename].append(yolo_format)

# 将每个文件的标注数据写入对应的 .txt 文件中
for filename, annotations in file_annotations.items():
    with open(os.path.join(output_dir, filename), 'w') as f:
        f.writelines(annotations)

print("YOLO格式转换完成！")

# 新功能：将.ppm格式图片转换为.png
input_image_dir = "/workspace/datasets/FullIJCNN2013"  # 假设这是存放.ppm图片的文件夹
output_image_dir = "/workspace/datasets/GTSRBYOLO"  # 这是转换后.png图片的目标文件夹
os.makedirs(output_image_dir, exist_ok=True)

# 读取指定文件夹中的所有.ppm图片并转换为.png
for image_file in os.listdir(input_image_dir):
    if image_file.endswith(".ppm"):
        ppm_path = os.path.join(input_image_dir, image_file)
        png_filename = image_file.replace('.ppm', '.png')
        png_path = os.path.join(output_image_dir, png_filename)

        # 使用Pillow打开并保存图像
        with Image.open(ppm_path) as img:
            img.save(png_path, 'PNG')

print("图片格式转换完成！")