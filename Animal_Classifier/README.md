基于ResNet18迁移学习的CNN分类器

数据集来源于网络，10分类任务

模型结构

Animal_Classifier/

├── checkpoints/       ✅ 断点续训、最佳模型存储

├── data/              ✅ 数据目录，含 train/val/test 子集

├── logs/              ✅ TensorBoard 日志输出

├── models/            ✅ 模型结构定义（ResNet18 + CBAM）

├── outputs/           🔄 用于导出 Excel、混淆矩阵图 等

├── scripts/           ✅ 主入口脚本（train.py, test_onnx.py）  可视化 混淆矩阵绘制

├── utils/

│   ├── train_eval.py  ✅ 训练 + 验证核心逻辑 + Tensor Board + 

│   ├── dataloader.py  ✅ 数据加载逻辑

│   ├── split_data.py    ✅ 数据集划分工具

│   └── export_onnx.py   ✅ ONNX部署推理

