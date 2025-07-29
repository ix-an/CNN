"""用于拆分未划分的数据集，训练集：测试集：验证集=8:1:1，适合小数据集"""
import os        # 文件操作，但是是"移动文件"，不能用于划分数据集
import shutil    # 用于复制文件，优势：原始文件保留，适用于数据集
import random


def split_dataset(src_dir, train_dir, val_dir, test_dir, split=(0.8,0.1,0.1)):
    """
    加载原始数据集，并拆分为训练集、测试集、验证集
    :param src_dir: 原始数据集路径
    :param train_dir: 训练集路径
    :param val_dir: 验证集路径
    :param test_dir: 测试集路径
    :param split: 训练集：测试集：验证集比例
    :return:None
    """
    assert sum(split) == 1.0, "划分比例总和必须为 1！"
    random.seed(22)    # 固定随机数种子，保证可复现

    # 创建目标目录（如果不存在）
    for dst_dir in [train_dir, val_dir, test_dir]:
        os.makedirs(dst_dir, exist_ok=True)

    # 遍历每个类别目录
    for class_name in os.listdir(src_dir):
        class_source = os.path.join(src_dir, class_name)
        if not os.path.isdir(class_source):
            continue    # 跳过非目录文件

        print(f"\n处理类别: {class_name}")

        # 为每个类别创建子目录，如 data/train/cat
        for dst_dir in [train_dir, test_dir, val_dir]:
            cls_dst = os.path.join(dst_dir, class_name)
            os.makedirs(cls_dst, exist_ok=True)

        # 获取当前类别下的所有图片
        images = os.listdir(class_source)
        # 打乱图片顺序
        random.shuffle(images)

        # 计算划分索引
        n = len(images)
        train_idx = int(n * split[0])
        val_idx = train_idx + int(n * split[1])

        # 拆分图片列表
        train_img = images[:train_idx]
        val_img = images[train_idx:val_idx]
        test_img = images[val_idx:]

        # 定义数据集类型、图片列表和目标目录的映射
        sets_info = [
            ("训练集", train_img, train_dir),
            ("验证集", val_img, val_dir),
            ("测试集", test_img, test_dir),
        ]

        # 统一处理每个数据集类型的复制操作
        for set_name, img_list, target_dir in sets_info:
            print(f"复制 {len(img_list)} 张图片到{set_name}...")
            for img in img_list:
                src = os.path.join(class_source, img)
                dst = os.path.join(target_dir, class_name, img)

                if not os.path.exists(src):
                    print(f"{set_name}划分过程中，发现源文件不存在: {src}")
                    continue

                shutil.copyfile(src, dst)

        print(f"类别 {class_name} 处理完成")

if __name__ == "__main__":
    source_path = "../data/animals"
    train_path = "../data/train"
    val_path = "../data/val"
    test_path = "../data/test"
    print("开始划分数据集")
    split_dataset(source_path, train_path, val_path, test_path)
    print("数据集划分完成")
