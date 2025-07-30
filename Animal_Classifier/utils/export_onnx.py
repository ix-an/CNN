"""导出ONNX模型"""
import torch
import torch.onnx as onnx
from models.pretrained_model import MyResNet18
import os

def export_onnx_model(pth_path, onnx_path, input_size=(1, 3, 224, 224)):
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型：与训练时一致
    model = MyResNet18(num_classes=10, freezy_layers=4, dropout_p=0.5)
    # 加载训练好的模型参数
    checkpoint = torch.load(pth_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # 创建虚拟输入
    dummy_input = torch.randn(input_size)

    # 导出ONNX模型
    torch.onnx.export(
        model, dummy_input, onnx_path,    # 模型、输入、输出路径
        input_names=['input'], output_names=['output'],  # 输入输出节点名称
        export_params=True,   # 导出模型参数
        opset_version=11,     # ONNX版本
        dynamic_axes={        # 动态轴配置：允许输入/输出的某些维度动态变化
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"✅ 导出成功：{onnx_path}")

if __name__ == '__main__':
    export_onnx_model(
        pth_path='../checkpoints/best_model.pth',
        onnx_path='../checkpoints/model.onnx'
    )
