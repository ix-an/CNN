"""使用ONNX模型进行推理"""
import onnxruntime as ort
import torch
import os
import numpy as np
from utils.datasets import get_dataloaders
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def softmax(x):
    """
    对输出的 logits 应用 softmax 归一化，得到概率分布。
    :param x: np.ndarray, 模型输出的 logits，形状为 (batch_size, num_classes)
    :return: np.ndarray, 归一化后的概率分布，形状与输入相同
    """
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # 防止溢出
    return e_x / e_x.sum(axis=1, keepdims=True)


def run_onnx_inference(onnx_path, test_dataloader):
    """
    使用 ONNX 模型对测试集进行推理，收集真实标签和预测标签。

    参数:
        onnx_path (str): ONNX 模型文件路径
        test_loader (DataLoader): 测试数据加载器

    返回:
        y_true (list): 真实标签
        y_pred (list): 预测标签
    """
    # 创建 ONNX 推理会话
    session = ort.InferenceSession(onnx_path)
    # 获取模型输入节点名称
    input_name = session.get_inputs()[0].name

    y_true = []    # 存储真实标签
    y_pred = []    # 存储预测标签

    for images, labels in test_dataloader:
        # 将tensor转换为ndarray
        np_imgs = images.numpy()
        # ONNX推理
        outputs = session.run(None, {input_name: np_imgs})[0]
        # 应用 softmax 归一化
        probs = softmax(outputs)
        # 获取预测类别：概率最大值的索引
        pred_labels = np.argmax(probs, axis=1)
        # 收集真实标签和预测标签
        y_true.extend(labels.numpy())
        y_pred.extend(pred_labels)

    return y_true, y_pred


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    绘制并保存混淆矩阵。

    参数:
        y_true (list): 真实标签
        y_pred (list): 预测标签
        class_names (list): 类别名称列表
        save_path (str, optional): 保存图像的路径（默认不保存）
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    # 创建混淆矩阵显示对象
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    # 绘制混淆矩阵
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Test Confusion Matrix")
    plt.tight_layout()    # 自动调整子图间距，防止标签重叠

    # 保存图像
    if save_path:
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)  # 保存图像
        print(f"✅ 混淆矩阵已保存至：{save_path}")

    plt.show()    # 显示图像


if __name__ == '__main__':
    _, _, test_loader = get_dataloaders(batch_size=32)
    true, pred = run_onnx_inference('../checkpoints/model.onnx', test_loader)
    class_names  = test_loader.dataset.classes
    plot_confusion_matrix(true, pred, class_names, '../outputs/confusion_matrix.png')
