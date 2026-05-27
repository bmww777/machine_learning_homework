# Severstal 钢铁表面缺陷检测 — 语义分割



本项目内置合成数据生成器，跳过 Kaggle 数据下载，直接验证完整训练 → 推理流程：

```bash
# 步骤 1: 极速训练 (合成数据, CPU 可用, 约 2~5 分钟)
python train.py --synthetic --epochs 10 --batch_size 2

# 步骤 2: 模型推理可视化 (自动加载 best_steel_unet.pth)
python demo_predict.py

# 步骤 3: 查看结果图
start demo_results.png
```

---

## 🔬 若有完整 Kaggle 数据集

将 `train_images/` 和 `train.csv` 放入 `data/severstal/` 后：

```bash
python train.py                                    # 启动真实数据训练
python demo_predict.py                             # 推理可视化
python plot_logs.py                                # 绘制 Loss/Dice 曲线
```

---

## 📂 项目结构

```
.
├── dataset.py          # 数据集: RLE 解码 + 数据增强
├── model.py            # 模型: UNetMultiBackbone (动态骨干网络)
├── losses.py           # 损失函数: BCE / Dice / BCEDiceLoss
├── train.py            # 训练入口: 支持消融实验 (见 --help)
├── plot_logs.py        # 可视化: 绘制实验 Loss/Dice 曲线
├── demo_predict.py     # 推理: 一键加载模型生成对比图
├── experiment_logs/    # 训练自动生成的 CSV 日志
├── demo_results.png    # 推理对比图输出
```

## 🖥 环境要求

| 包 | 版本 | 安装 |
|---|------|------|
| Python | ≥ 3.8 | — |
| PyTorch + torchvision | ≥ 2.0 | `pip install torch torchvision` |
| timm | ≥ 0.9.0 | `pip install timm` |
| albumentations | ≥ 1.3.0 | `pip install albumentations` |
| opencv-python | ≥ 4.5 | `pip install opencv-python` |
| pandas, numpy, matplotlib | 最新 | `pip install pandas numpy matplotlib` |

---

## 🧪 消融实验快速命令

```bash
python train.py --backbone resnet18 --loss bce_dice --synthetic --epochs 20
python train.py --backbone resnet50 --loss bce_dice --synthetic --epochs 20
python train.py --backbone resnet34 --loss bce      --synthetic --epochs 20
python train.py --backbone resnet34 --loss dice     --synthetic --epochs 20

# 实验结束后，一键生成对比曲线
python plot_logs.py
```

---

> 注：真实 Severstal 数据集约 12 GB。使用 `--synthetic` 参数可在无数据时快速运行验证。
