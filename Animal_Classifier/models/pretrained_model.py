"""
使用ResNet18迁移学习模型，添加CBAM注意力机制，BN批量标准化，Dropout正则化，
并修改全连接层适配10分类任务
Input Image →
→ ResNet18 卷积层 (冻结前几层)
→ Layer4（后几层，仍在训练）
→ CBAM 注意力机制
→ AdaptiveAvgPool + Flatten
→ BatchNorm1d + Dropout
→ Linear (输出10类)
→ Softmax/Logits
"""
"""ResNet18结构
[conv1]       --> 卷积
[bn1]
[relu]
[maxpool]

[layer1]
[layer2]
[layer3]
[layer4]      <-- 到这为止都是提取特征
[avgpool]     --> 原始的分类前池化
[fc]          --> 原始的1000类全连接层

"""

import torch
import torch.nn as nn
from torchvision.models import resnet18,ResNet18_Weights

# CBAM 注意力机制
class CBAM(nn.Module):
    def __init__(self, in_planes, reduction=16, kernel_size=7):
        super().__init__()
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // reduction, in_planes, kernel_size=1, bias=False),
        )
        # 空间注意力
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        # sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道注意力
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_att

        return x


# 定义模型：基于ResNet18
class MyResNet18(nn.Module):
    def __init__(self, num_classes=10, freezy_layers=4, dropout_p=0.5):
        super(MyResNet18, self).__init__()

        # 加载预训练模型
        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # 冻结前几层
        layers = list(base_model.children())
        for idx, layer in enumerate(layers[:freezy_layers]):
            for param in layer.parameters():
                param.requires_grad = False

        # 提取特征部分：去掉原模型的fc
        self.features = nn.Sequential(*layers[:2])    # 抽出所有卷积层部分
        # 添加CBAM注意力机制
        self.cbam = CBAM(512)    # ResNet18最后一个block输出通道数是512

        # 自定义全连接层（输出头）
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_p),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)      # 提取出的高阶特征图
        x = self.cbam(x)          # 用CBAM加强重点区域
        x = self.classifier(x)    # 池化+分类（全连接层）
        return x


if __name__ == '__main__':
    model = MyResNet18()
    print(model)







