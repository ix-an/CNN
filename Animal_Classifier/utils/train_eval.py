"""训练+验证方法，包含TensorBoard"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm


"""模型训练主流程"""
def train_model(
        model,
        train_loader,
        val_loader,
        num_epochs,
        device,
        save_path='../checkpoints/checkpoint.pth',    # 模型保存路径（.pth 文件）
        resume=False,    # 是否从断点（保存的模型）恢复训练（bool）
        log_dir='../logs',    # TensorBoard 日志保存目录
        lr=0.001,
):
    # ------------------------------------------
    # 1. 模型准备
    # ------------------------------------------
    model = model.to(device)

    # Adam优化器自带 动量法 + 自适应学习率
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()

    # 学习率调度器：每 10轮 lr×0.1
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # TensorBoard 初始化
    writer = SummaryWriter(log_dir=log_dir)

    # 初始化最佳验证准确率和起始轮次
    best_acc = 0.0
    start_epoch = 0

    # ------------------------------------------
    # 2. 断点续训
    # ------------------------------------------
    if resume and os.path.exists(save_path):
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model'])            # 加载模型参数
        optimizer.load_state_dict(checkpoint['optimizer'])    # 加载优化器状态
        scheduler.load_state_dict(checkpoint['scheduler'])    # 加载调度器状态
        start_epoch = checkpoint['epoch'] + 1                 # 从下一轮开始
        best_acc = checkpoint['best_acc']                     # 加载历史最佳准确率
        print(f"✅ Resumed from epoch {start_epoch}, best val acc = {best_acc:.4f}")
    else:
        print("📦 Starting first-time training: loading pretrained ResNet18 weights")
        from torchvision.models import resnet18, ResNet18_Weights
        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.load_state_dict(base_model.state_dict(), strict=False)

    # ------------------------------------------
    # 3. 训练主循环
    # ------------------------------------------
    for epoch in range(start_epoch, num_epochs):
        model.train()    # 训练模式
        running_loss = 0.0
        correct = 0    # 当前批次预测正确的样本数
        total = 0      # 当前批次的总样本数

        # 使用tqdm显示训练进度条
        loop = tqdm(train_loader, desc=f'[Epoch {epoch+1}/{num_epochs}]')

        for i, (images, labels) in enumerate(loop):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()                # 清空梯度
            outputs = model(images)              # 前向传播
            loss = criterion(outputs, labels)    # 计算损失
            loss.backward()                      # 反向传播
            optimizer.step()                     # 更新参数

            # 统计训练信息
            _, preds = torch.max(outputs.data, 1)    # 获取预测结果
            running_loss += loss.item()              # 累加损失
            correct += (preds==labels).sum().item()    # 累加正确预测数
            total += labels.size(0)                  # 累加总样本数

            # 在进度条中显示当前损失和准确率
            loop.set_postfix(loss=loss.item(), acc=correct/total)

        train_loss = running_loss / len(train_loader)    # 平均训练损失
        train_acc = correct / total                      # 平均训练准确率

        # ----------------------------------
        # 4. 验证阶段：使用验证函数
        # ----------------------------------
        val_loss, val_acc, _, _ = validate_model(model, val_loader, criterion, device)

        # ----------------------------------
        # 5. TensorBoard记录
        # ----------------------------------
        # 记录标量（损失、准确率）
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Acc/train', train_acc, epoch)
        writer.add_scalar('Acc/val', val_acc, epoch)
        writer.add_scalars('Compare', {
            'train_acc': train_acc,
            'val_acc': val_acc
        }, epoch)
        # 记录模型参数直方图
        writer.add_histogram('fc_weights', model.classifier[-1].weight, epoch)
        # 在第一次记录模型结构
        if epoch == 0:
            writer.add_graph(model, images)

        # -----------------------------------
        # 6. 保存模型
        # -----------------------------------
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),            # 模型参数
                'optimizer': optimizer.state_dict(),    # 优化器状态
                'scheduler': scheduler.state_dict(),    # 调度器状态
                'best_acc': best_acc                    # 最佳准确率
            }, save_path)
            print(f"🌟Saved new best model at epoch {epoch+1} (acc={best_acc:.4f})")

        # 更新学习率
        scheduler.step()

    writer.close()


"""模型验证集方法"""
def validate_model(model, val_loader, criterion, device):
    model.eval()        # 评估模式
    total = 0           # 总样本数
    correct = 0         # 正确预测的样本数
    loss_total = 0.0    # 总损失

    # 用于生成报表
    y_true_all = []    # 记录所有真实标签
    y_pred_all = []    # 记录所有预测标签

    with torch.no_grad():    # 关闭梯度计算
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)    # 计算损失
            loss_total += loss.item()

            _, preds = torch.max(outputs, 1)    # 获取预测结果
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # 🍬 注意需要将标签从GPU移动到CPU，再转换为numpy数组
            y_true_all.extend(labels.cpu().numpy())    # 添加真实标签
            y_pred_all.extend(preds.cpu().numpy())

    avg_loss = loss_total / len(val_loader)    # 验证集平均损失
    avg_acc = correct / total                  # 验证集平均准确率

    print(f"Validation -> Loss:{avg_loss:.4f} | Acc:{avg_acc:.4f}")
    return avg_loss, avg_acc, y_true_all, y_pred_all