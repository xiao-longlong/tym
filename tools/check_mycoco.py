import os

def find_files(directory, extension):
    """
    在指定目录下查找所有具有特定扩展名的文件
    """
    files_with_extension = set()
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extension):
                files_with_extension.add(os.path.join(root, file))
    return files_with_extension

def check_files(png_directory, txt_directory):
    # 查找所有 .png 和 .txt 文件
    png_files = find_files(png_directory, '.png')
    txt_files = find_files(txt_directory, '.txt')

    # 获取 .png 文件名 (不包括扩展名) 和 .txt 文件名 (不包括扩展名)
    png_base_names = {os.path.splitext(os.path.basename(file))[0] for file in png_files}
    txt_base_names = {os.path.splitext(os.path.basename(file))[0] for file in txt_files}

    # 查找 .png 文件没有对应的 .txt 文件
    png_without_txt = png_base_names - txt_base_names
    if png_without_txt:
        print("PNG files without corresponding TXT files:")
        for root, _, files in os.walk(png_directory):
            for file in files:
                if file.lower().endswith('.png'):
                    base_name = os.path.splitext(file)[0]
                    if base_name in png_without_txt:
                        print(os.path.join(root, file))

    # 查找 .txt 文件没有对应的 .png 文件
    txt_without_png = txt_base_names - png_base_names
    if txt_without_png:
        print("TXT files without corresponding PNG files:")
        for root, _, files in os.walk(txt_directory):
            for file in files:
                if file.lower().endswith('.txt'):
                    base_name = os.path.splitext(file)[0]
                    if base_name in txt_without_png:
                        print(os.path.join(root, file))

# 示例目录
png_directory = '/workspace/datasets/GTSRBYOLO2/images'
txt_directory = '/workspace/datasets/GTSRBYOLO2/labels'

check_files(png_directory, txt_directory)
