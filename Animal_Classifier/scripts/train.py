"""
训练主入口：调用模型、数据加载、配置训练参数、训练
"""
import torch
import os
from models.pretrained_model import MyResNet18    # 自定义的迁移学习网络结构
from utils.train_eval import train_model, validate_model  # 自定义训练函数和验证函数
from utils.datasets import get_dataloaders        # 自定义数据加载函数
import pandas as pd
from sklearn.metrics import classification_report


"""训练入口函数"""
def main():
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🖥️ using device:", device)

    # 是否启用断点续训
    resume = True

    # 训练轮次和学习率
    epochs = 25
    lr = 0.001

    # 构建模型
    model = MyResNet18(num_classes=10, freezy_layers=4, dropout_p=0.5)
    save_path = '../checkpoints/best_model.pth'

    # 数据加载器
    train_loader, val_loader, _ = get_dataloaders(batch_size=32)

    # 启动训练
    train_model(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs,
        device=device,
        save_path=save_path,
        resume=resume,
        log_dir='../logs',
        lr=lr
    )

    # 训练完成后，加载最佳模型，进行验证报表分析
    print("\n 训练完成，开始加载最佳模型并进行验证分析...")
    validation_report(model, val_loader, torch.nn.CrossEntropyLoss(), device, save_path)


"""验证集报表分析函数"""
def validation_report(model, val_loader, criterion, device, save_path):
    try:
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        model = model.to(device)

        _, _, y_true_all, y_pred_all = validate_model(model, val_loader, criterion, device)

        # 生成分类报表和 Excel文件
        report_dict = classification_report(y_true_all, y_pred_all, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        os.makedirs('../outputs', exist_ok=True)
        report_df.to_excel('../outputs/val_report.xlsx')
        print("验证集分类表格已保存到: ../outputs/val_report.xlsx")
        print(report_df)
        print(f"📊 验证集分类精确率：{report_df.loc['weighted avg', 'precision']:.4f}")
        print(f"📊 验证集分类召回率：{report_df.loc['weighted avg', 'recall']:.4f}")
        print(f"📊 验证集分类F1分数：{report_df.loc['weighted avg', 'f1-score']:.4f}")
    except Exception as e:
        print(f"生成报表时出现错误: {e}")



if __name__ == '__main__':
    main()