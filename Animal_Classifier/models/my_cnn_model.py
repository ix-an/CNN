"""用于动物分类的模型：10个动物
['butterfly','cat','dog','dolphin','eagle','flamingo','jellyfish','panda','tiger','zebra']
"""

import torch
import torch.nn as nn


class AnimalClassifier(nn.Module):
    def __init__(self, num_classes=10, input_size=224):
        super(AnimalClassifier, self).__init__()
        # 卷积层
        self.features = nn.Sequential(
            # 第一层 卷积 + BN + 最大池化 + ReLu
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),    # 原地操作，节省内存
            # 第二层 卷积 + BN + 最大池化 + ReLu
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),
            # 第三层 卷积 + BN + 最大池化 + ReLu
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),
            # 第四层 卷积 + BN + 最大池化 + ReLu
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),
        )
        # 计算全连接层的输入维数
        final_size = input_size // (2 ** 4)    # 经过4次最大池化，图片尺寸缩小为1/16
        linear_in = 128 * final_size * final_size
        # 全连接层
        self.classifier = nn.Sequential(
            nn.Linear(linear_in, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # x[B,C,H,W] -> x[B,C*H*W]
        x = self.classifier(x)
        return x

