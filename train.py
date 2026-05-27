"""

【消融实验设计】
本脚本通过顶部的 CONFIG 字典实现快速切换实验配置，不需修改核心代码:
  - 骨干网络消融: 修改 CONFIG['backbone'] 切换不同 Encoder。
  - 损失函数消融: 修改 CONFIG['loss_type'] 与 CONFIG['bce_weight']。
  - 单次运行即自动记录不同配置下的训练指标，便于后续定量对比分析。

运行方式:
  python train.py                     # 使用默认配置训练
  python train.py --synthetic         # 使用合成数据快速验证 pipeline
  python train.py --backbone resnet50 # 命令行覆盖骨干网络
  python train.py --loss bce          # 命令行覆盖损失函数
"""

import os
import sys
import math
import random
import argparse
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp
from torch.utils.data import Dataset, DataLoader

from dataset import SteelDataset, get_train_transforms, get_val_transforms
from model import UNetMultiBackbone
from losses import get_loss


CONFIG = {
    # ── 数据 ──
    'data_dir': './data/severstal',
    'image_dir': './data/severstal/train_images',
    'csv_path': './data/severstal/train.csv',
    'image_size': (256, 1600),

    # ── 模型 ──
    # 可选: resnet18, resnet34, resnet50, resnet101, efficientnet_b0, efficientnet_b4
    'backbone': 'resnet34',
    'pretrained': True,
    'num_classes': 1,  # 二分类语义分割

    # ── 损失函数 ──
    # 可选: 'bce', 'dice', 'bce_dice'
    'loss_type': 'bce_dice',
    'bce_weight': 0.5,    # BCE 在混合损失中的权重 (仅 loss_type='bce_dice' 时生效)
    'pos_weight': 2.0,    # BCE 正样本权重

    # ── 优化器 ──
    'optimizer': 'adam',
    'learning_rate': 3e-4,
    'weight_decay': 1e-5,

    # ── 学习率调度 ──
    'lr_scheduler': 'plateau',
    'lr_factor': 0.5,          # ReduceLROnPlateau 的衰减因子
    'lr_patience': 5,          # 验证指标停滞多少个 epoch 后降低学习率

    # ── 训练超参数 ──
    'batch_size': 32,
    'num_epochs': 50,
    'num_workers': 0,          # Windows 下建议设为 0 避免多进程问题
    'train_split': 0.8,        # 训练集比例 (剩余为验证集)

    # ── 硬件 ──
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # ── 输出 ──
    'save_path': './best_steel_unet.pth',
    'log_interval': 20,        # 每 N 个 batch 打印一次训练日志
}


# ═══════════════════════════════════════════════════════════════════════════════
# 合成数据生成 (用于快速验证 Pipeline，无需下载 Kaggle 数据)
# ═══════════════════════════════════════════════════════════════════════════════

class SyntheticSteelDataset(Dataset):
    """合成钢铁缺陷数据集，用于在无 Kaggle 数据时快速验证训练流程。

    生成 256×1600 的纯色背景图像，随机放置 1~5 个矩形"缺陷"区域。
    Mask 为对应的二值标签。
    """

    def __init__(self, num_samples=200, image_size=(256, 1600), transforms=None):
        self.num_samples = num_samples
        self.image_size = image_size
        self.transforms = transforms

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        H, W = self.image_size

        # 生成随机灰度背景 (模拟钢材表面)
        base_intensity = random.randint(100, 180)
        image = np.full((H, W, 3), base_intensity, dtype=np.uint8)

        # 添加纹理噪声
        noise = np.random.randint(-15, 15, (H, W, 3), dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # 生成随机缺陷 Mask
        mask = np.zeros((H, W), dtype=np.float32)
        num_defects = random.randint(1, 5)

        for _ in range(num_defects):
            # 随机缺陷位置与尺寸
            defect_h = random.randint(5, 30)
            defect_w = random.randint(10, 80)
            y = random.randint(0, H - defect_h)
            x = random.randint(0, W - defect_w)

            mask[y:y + defect_h, x:x + defect_w] = 1.0

            # 在缺陷区域改变图像颜色 (模拟表面异常)
            image[y:y + defect_h, x:x + defect_w, :] = \
                np.random.randint(30, 80, (defect_h, defect_w, 3), dtype=np.uint8)

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0)

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        image_id = f"synthetic_{idx:04d}"
        return image, mask, image_id


# ═══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed=42):
    """固定随机种子以确保实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_log_filename(config):

    backbone = config['backbone']
    loss_type = config['loss_type']
    parts = ['log', backbone, loss_type]

    # BCE 权重仅在混合损失且非默认 0.5 时附加到文件名
    if loss_type == 'bce_dice':
        w = config.get('bce_weight', 0.5)
        if abs(w - 0.5) > 1e-4:
            parts.append(f'wbce{w}')

    return '_'.join(parts) + '.csv'


def compute_dice(pred, target, smooth=1.0):

    pred_flat = pred.contiguous().view(pred.shape[0], -1)
    target_flat = target.contiguous().view(target.shape[0], -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

    dice_per_sample = (2.0 * intersection + smooth) / (union + smooth)
    return dice_per_sample.mean().item()


# ═══════════════════════════════════════════════════════════════════════════════
# 训练一个 Epoch
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch,
                    log_interval=20, scaler=None):

    model.train()
    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0
    num_batches = len(dataloader)

    for batch_idx, (images, masks, _) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # ── 前向传播 (混合精度) ──
        with torch.cuda.amp.autocast():
            logits = model(images)
            loss_output = criterion(logits, masks)

            # 统一处理返回格式: 兼容返回标量或 dict 的损失函数
            if isinstance(loss_output, dict):
                loss = loss_output['loss']
                bce_val = loss_output.get('bce_loss', None)
                dice_val = loss_output.get('dice_loss', None)
            else:
                loss = loss_output
                bce_val = None
                dice_val = None

        # ── 反向传播 (梯度缩放) ──
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # ── 统计 ──
        total_loss += loss.item()
        if bce_val is not None:
            total_bce += bce_val.item()
        if dice_val is not None:
            total_dice += dice_val.item()

        # ── 日志 ──
        if (batch_idx + 1) % log_interval == 0 or batch_idx == 0:
            msg = f"  Epoch {epoch:3d} | Batch {batch_idx+1:4d}/{num_batches} | Loss: {loss.item():.4f}"
            if bce_val is not None and dice_val is not None:
                msg += f" | BCE: {bce_val.item():.4f} | Dice: {dice_val.item():.4f}"
            print(msg)

    return {
        'loss': total_loss / num_batches,
        'bce': total_bce / num_batches if total_bce > 0 else None,
        'dice': total_dice / num_batches if total_dice > 0 else None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 验证一个 Epoch
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(model, dataloader, criterion, device):

    model.eval()
    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0
    total_dice_coeff = 0.0
    num_batches = len(dataloader)

    for images, masks, _ in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        # ── 前向传播 (混合精度推理，节省显存) ──
        with torch.cuda.amp.autocast():
            logits = model(images)
            loss_output = criterion(logits, masks)

        if isinstance(loss_output, dict):
            loss = loss_output['loss']
            bce_val = loss_output.get('bce_loss', None)
            dice_val = loss_output.get('dice_loss', None)
        else:
            loss = loss_output
            bce_val = None
            dice_val = None

        total_loss += loss.item()
        if bce_val is not None:
            total_bce += bce_val.item()
        if dice_val is not None:
            total_dice += dice_val.item()

        # ── 计算 Dice 系数 (使用阈值 0.5 二值化预测) ──
        pred_binary = (torch.sigmoid(logits) > 0.5).float()
        total_dice_coeff += compute_dice(pred_binary, masks)

    return {
        'loss': total_loss / num_batches,
        'bce': total_bce / num_batches if total_bce > 0 else None,
        'dice': total_dice / num_batches if total_dice > 0 else None,
        'dice_coeff': total_dice_coeff / num_batches,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 主训练函数
# ═══════════════════════════════════════════════════════════════════════════════

def main(config):

    set_seed(42)

    # ──────────────────────────────────────────────────────────────────────
    # 1. 准备数据
    # ──────────────────────────────────────────────────────────────────────
    use_synthetic = config.get('use_synthetic', False)
    csv_path = config['csv_path']

    if use_synthetic or not os.path.exists(csv_path):
        if not use_synthetic:
            print(f"⚠ 未找到 CSV 文件: {csv_path}")
            print("  自动回退到合成数据模式。如需使用真实数据，请修改 CONFIG['csv_path']。\n")
        print("=" * 70)
        print("使用合成数据集进行训练")
        print("=" * 70)

        total_samples = 200
        # 生成所有样本索引的随机排列
        all_indices = list(range(total_samples))
        random.shuffle(all_indices)
        split = int(total_samples * config['train_split'])
        train_indices = all_indices[:split]
        val_indices = all_indices[split:]

        # 训练集: 带数据增强
        train_dataset = SyntheticSteelDataset(
            num_samples=len(train_indices),
            image_size=config['image_size'],
            transforms=get_train_transforms(config['image_size']),
        )
        # 验证集: 仅归一化，无数据增强
        val_dataset = SyntheticSteelDataset(
            num_samples=len(val_indices),
            image_size=config['image_size'],
            transforms=get_val_transforms(config['image_size']),
        )
    else:
        print("=" * 70)
        print(f"加载数据集: {csv_path}")
        print("=" * 70)
        df = pd.read_csv(csv_path)
        print(f"  CSV 记录数: {len(df)}")
        print(f"  唯一图像数: {df['ImageId'].nunique()}")

        # 按 ImageId 去重后划分训练/验证集，确保同一图像的不同缺陷类别不跨分
        unique_ids = df['ImageId'].unique().tolist()
        random.shuffle(unique_ids)
        split = int(len(unique_ids) * config['train_split'])
        train_ids = set(unique_ids[:split])
        val_ids = set(unique_ids[split:])

        # 分别构建训练与验证 DataFrame
        train_df = df[df['ImageId'].isin(train_ids)].reset_index(drop=True)
        val_df = df[df['ImageId'].isin(val_ids)].reset_index(drop=True)

        # 训练集: 带数据增强
        train_dataset = SteelDataset(
            df=train_df,
            image_dir=config['image_dir'],
            transforms=get_train_transforms(config['image_size']),
        )
        # 验证集: 仅归一化，无数据增强 (保证评估一致性)
        val_dataset = SteelDataset(
            df=val_df,
            image_dir=config['image_dir'],
            transforms=get_val_transforms(config['image_size']),
        )

    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")

    # ──────────────────────────────────────────────────────────────────────
    # 3. DataLoader
    # ──────────────────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=(config['device'] == 'cuda'),
        drop_last=True,  # 丢弃不完整 batch，避免 BatchNorm 报错
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=(config['device'] == 'cuda'),
    )

    # ──────────────────────────────────────────────────────────────────────
    # 4. 模型初始化
    # ──────────────────────────────────────────────────────────────────────
    device = torch.device(config['device'])
    print(f"\n使用设备: {device}")
    if device.type == 'cuda':
        print(f"  GPU 型号: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA 版本: {torch.version.cuda}")

    print(f"\n初始化模型 (骨干网络: {config['backbone']})...")
    model = UNetMultiBackbone(
        backbone_name=config['backbone'],
        pretrained=config['pretrained'],
        num_classes=config['num_classes'],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量:    {total_params / 1e6:.2f}M")
    print(f"  可训练参数:  {trainable_params / 1e6:.2f}M")
    print(f"  Encoder 通道: {model.get_encoder_channels()}")

    # ── 混合精度训练 ──
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print(f"\n混合精度训练 (AMP) 已启用 — FP16 前向 + GradScaler 反向")

    # ──────────────────────────────────────────────────────────────────────
    # 5. 损失函数
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n损失函数配置:")
    print(f"  类型: {config['loss_type']}")
    if config['loss_type'] == 'bce_dice':
        print(f"  BCE 权重 (w_bce): {config['bce_weight']}")
        print(f"  Dice 权重: {1 - config['bce_weight']}")
    print(f"  BCE 正样本权重: {config['pos_weight']}")

    criterion = get_loss(
        config['loss_type'],
        w_bce=config['bce_weight'],
        pos_weight=config['pos_weight'],
    )

    # ──────────────────────────────────────────────────────────────────────
    # 6. 优化器
    # ──────────────────────────────────────────────────────────────────────
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )
    print(f"\n优化器: Adam (lr={config['learning_rate']}, wd={config['weight_decay']})")

    # ──────────────────────────────────────────────────────────────────────
    # 7. 学习率调度器
    # ──────────────────────────────────────────────────────────────────────

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['lr_factor'],
        patience=config['lr_patience'],
    )
    print(f"学习率调度: ReduceLROnPlateau (factor={config['lr_factor']}, "
          f"patience={config['lr_patience']})")

    # ──────────────────────────────────────────────────────────────────────
    # 8. 实验日志初始化
    # ──────────────────────────────────────────────────────────────────────
    log_dir = Path('experiment_logs')
    log_dir.mkdir(exist_ok=True)
    log_filename = build_log_filename(config)
    log_path = log_dir / log_filename

    # 用于收集每个 Epoch 的指标记录
    log_records = []

    # ──────────────────────────────────────────────────────────────────────
    # 9. 训练循环
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("开始训练")
    print("=" * 70)
    print(f"  实验日志将保存至: {log_path}")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>10} | "
          f"{'Val Dice':>9} | {'LR':>10} | Best")
    print("-" * 70)

    best_val_dice = 0.0
    best_epoch = 0
    save_path = config['save_path']

    try:
        for epoch in range(1, config['num_epochs'] + 1):
            current_lr = optimizer.param_groups[0]['lr']

            # ── 训练 ──
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                epoch=epoch, log_interval=config['log_interval'],
                scaler=scaler,
            )

            # ── 验证 ──
            val_metrics = validate(model, val_loader, criterion, device)

            # ── 学习率调整 ──
            scheduler.step(val_metrics['loss'])

            # ── 收集本 Epoch 的实验指标 ──
            log_records.append({
                'epoch': epoch,
                'train_loss': round(train_metrics['loss'], 6),
                'val_loss': round(val_metrics['loss'], 6),
                'val_dice': round(val_metrics['dice_coeff'], 6),
                'lr': current_lr,
            })

            # ── 保存最佳模型 (以验证集 Dice 系数为基准) ──
            is_best = val_metrics['dice_coeff'] > best_val_dice
            if is_best:
                best_val_dice = val_metrics['dice_coeff']
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_dice': best_val_dice,
                    'config': {k: v for k, v in config.items()
                               if isinstance(v, (str, int, float, bool, list, tuple))},
                }, save_path)

            # ── 打印 Epoch 摘要 ──
            best_marker = "*" if is_best else ""
            print(f"{epoch:4d}  | {train_metrics['loss']:10.4f} | "
                  f"{val_metrics['loss']:10.4f} | "
                  f"{val_metrics['dice_coeff']:9.4f} | "
                  f"{current_lr:10.2e} | {best_marker:>5}")

    except KeyboardInterrupt:
        print("\n⚠  训练被用户中断，正在保存已收集的实验日志...")
    finally:
        # ── 保存实验日志 CSV (无论正常结束还是中断均执行) ──
        if log_records:
            df_log = pd.DataFrame(log_records)
            df_log.to_csv(log_path, index=False, encoding='utf-8-sig')
            print(f"\n📊 实验日志已保存至: {os.path.abspath(log_path)}")
            print(f"   共 {len(log_records)} 条 Epoch 记录, 列: {list(df_log.columns)}")

    # ──────────────────────────────────────────────────────────────────────
    # 10. 训练完成摘要
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)
    print(f"  最佳验证 Dice: {best_val_dice:.4f} (Epoch {best_epoch})")
    print(f"  最佳模型已保存至: {os.path.abspath(save_path)}")
    print(f"\n消融实验配置记录:")
    print(f"  骨干网络:     {config['backbone']}")
    print(f"  损失函数:     {config['loss_type']}")
    if config['loss_type'] == 'bce_dice':
        print(f"  BCE 权重:    {config['bce_weight']}")
    print(f"  学习率:       {config['learning_rate']}")
    print(f"  Batch Size:   {config['batch_size']}")
    print(f"  Epochs:       {config['num_epochs']}")
    print("=" * 70)

    return best_val_dice


# ═══════════════════════════════════════════════════════════════════════════════
# 命令行入口
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(
        description="Severstal 钢铁缺陷语义分割 — 训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
消融实验示例:
  python train.py --backbone resnet18 --loss bce_dice --bce_weight 0.5
  python train.py --backbone resnet50 --loss bce_dice --bce_weight 0.5
  python train.py --backbone resnet34 --loss bce      # 纯 BCE 实验
  python train.py --backbone resnet34 --loss dice     # 纯 Dice 实验
  python train.py --synthetic                          # 使用合成数据快速测试
        """,
    )
    parser.add_argument('--backbone', type=str, default=None,
                        help='骨干网络名称 (默认: resnet34)')
    parser.add_argument('--loss', type=str, default=None,
                        choices=['bce', 'dice', 'bce_dice'],
                        help='损失函数类型')
    parser.add_argument('--bce_weight', type=float, default=None,
                        help='BCE 在混合损失中的权重 ∈ [0, 1]')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=None,
                        help='初始学习率')
    parser.add_argument('--synthetic', action='store_true',
                        help='使用合成数据 (无需 Kaggle 数据集)')
    parser.add_argument('--save_path', type=str, default=None,
                        help='模型保存路径')

    args = parser.parse_args()

    # 用命令行参数覆盖 CONFIG
    if args.backbone is not None:
        CONFIG['backbone'] = args.backbone
    if args.loss is not None:
        CONFIG['loss_type'] = args.loss
    if args.bce_weight is not None:
        CONFIG['bce_weight'] = args.bce_weight
    if args.epochs is not None:
        CONFIG['num_epochs'] = args.epochs
    if args.batch_size is not None:
        CONFIG['batch_size'] = args.batch_size
    if args.lr is not None:
        CONFIG['learning_rate'] = args.lr
    if args.save_path is not None:
        CONFIG['save_path'] = args.save_path
    if args.synthetic:
        CONFIG['use_synthetic'] = True

    main(CONFIG)
