import os
import shutil
import random
def get_prefix_before_recon(input_dir):
    prefixes = []
    # 遍历文件夹中的所有文件
    for filename in os.listdir(input_dir):
        # 检查文件名中是否包含 "recon"
        if "recon" in filename:
            # 找到 "recon" 的位置
            recon_index = filename.index("recon")
            # 提取 "recon" 之前的字符
            prefix = filename[:recon_index]

            # 如果前缀不为空，则添加到列表中
            if prefix:
                prefixes.append(prefix)
    return prefixes
def split_prefixes(prefixes, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # 去重
    unique_prefixes = list(set(prefixes))
    # 打乱顺序
    random.shuffle(unique_prefixes)
    # 计算划分点
    total = len(unique_prefixes)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    # 划分数据集
    train_set = unique_prefixes[:train_count]
    val_set = unique_prefixes[train_count:train_count + val_count]
    test_set = unique_prefixes[train_count + val_count:]
    return train_set, val_set, test_set

def write_to_file(filename, prefixes):
    with open(filename, 'w') as f:
        for prefix in prefixes:
            f.write(prefix + '\n')

def move_files_by_prefix(input_dir, train_set, val_set, test_set, train_dir, val_dir, test_dir):
    # 创建目标目录
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 遍历文件夹中的所有文件
    for filename in os.listdir(input_dir):
        # 检查文件名中是否包含 "recon"
        if "recon" in filename:
            # 找到 "recon" 的位置
            recon_index = filename.index("recon")
            # 提取 "recon" 之前的字符
            prefix = filename[:recon_index]
            # 根据前缀移动文件
            if prefix in train_set:
                shutil.move(os.path.join(input_dir, filename), os.path.join(train_dir, filename))
                print(f"Moved {filename} to {train_dir}")
            elif prefix in val_set:
                shutil.move(os.path.join(input_dir, filename), os.path.join(val_dir, filename))
                print(f"Moved {filename} to {val_dir}")
            elif prefix in test_set:
                shutil.move(os.path.join(input_dir, filename), os.path.join(test_dir, filename))
                print(f"Moved {filename} to {test_dir}")
if __name__ == "__main__":
    # 输入文件夹路径
    input_directory = "realdata/label"
    prefixes = get_prefix_before_recon(input_directory)
    # 划分数据集
    train_set, val_set, test_set = split_prefixes(prefixes)

    train_directory = "train/label"
    val_directory = "val/label"
    test_directory = "test/label"
    move_files_by_prefix("realdata/label", train_set, val_set, test_set, train_directory, val_directory, test_directory)

    #######################################
    train_directory = "train/low"
    val_directory = "val/low"
    test_directory = "test/low"
    move_files_by_prefix("realdata/low", train_set, val_set, test_set, train_directory, val_directory, test_directory)

    # 输出到文件
    write_to_file("train_prefixes.txt", train_set)
    write_to_file("val_prefixes.txt", val_set)
    write_to_file("test_prefixes.txt", test_set)
