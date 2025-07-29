"""数据加载和预处理"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 定义预处理配置
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomResizedCrop(224),  # 随机裁剪+缩放
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    transforms.ToTensor(),  # 转为 Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 归一化
])

VAL_TEST_TRANSFORMS = transforms.Compose([
    transforms.CenterCrop(224),  # 中心裁剪到 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_dataloaders(batch_size : int = 32):
    """
    获取训练、验证、测试数据集的DataLoader
    :param batch_size:批次大小
    :return:train_loader, val_loader, test_loader
    """
    # 加载数据集
    train_dataset = datasets.ImageFolder(root="../data/train", transform=TRAIN_TRANSFORMS)
    val_dataset = datasets.ImageFolder(root="../data/val", transform=VAL_TEST_TRANSFORMS)
    test_dataset = datasets.ImageFolder(root="../data/test", transform=VAL_TEST_TRANSFORMS)

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # 测试数据集能否正常加载
    train, val, test = get_dataloaders()
    print(train.dataset)
    print(val.dataset)
    print(test.dataset)
