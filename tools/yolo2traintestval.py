import os
import shutil
import random

def split_dataset(dataset_path, output_path, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    # 检查输出文件夹是否存在，不存在则创建
    images_output_dir = os.path.join(output_path, 'images')
    annotations_output_dir = os.path.join(output_path, 'annotations')

    for subdir in ['train', 'val', 'test']:
        os.makedirs(os.path.join(images_output_dir, subdir), exist_ok=True)
        os.makedirs(os.path.join(annotations_output_dir, subdir), exist_ok=True)

    # 获取所有图片和标注文件
    image_files = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.png'))]
    annotation_files = [f.replace('.jpg', '.txt').replace('.png', '.txt') for f in image_files]

    # 确保图片和标注文件一一对应
    dataset = list(zip(image_files, annotation_files))

    # 随机打乱数据集
    random.shuffle(dataset)

    # 划分数据集
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size + val_size]
    test_dataset = dataset[train_size + val_size:]

    # 将数据集按比例分配到相应的文件夹中
    for subset, subset_name in [(train_dataset, 'train'), (val_dataset, 'val'), (test_dataset, 'test')]:
        for image_file, annotation_file in subset:
            # 移动图片文件
            src_image = os.path.join(dataset_path, image_file)
            dst_image = os.path.join(images_output_dir, subset_name, image_file)
            shutil.copyfile(src_image, dst_image)

            # 移动标注文件
            src_annotation = os.path.join(dataset_path, annotation_file)
            if os.path.exists(src_annotation):
                dst_annotation = os.path.join(annotations_output_dir, subset_name, annotation_file)
                shutil.copyfile(src_annotation, dst_annotation)

if __name__ == "__main__":
    dataset_path = '/workspace/datasets/GTSRBYOLO'  # 替换为你的数据集路径
    output_path = '/workspace/datasets/GTSRBYOLO2'  # 替换为你希望保存结果的路径
    split_dataset(dataset_path, output_path, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
