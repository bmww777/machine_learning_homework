

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# Dice Loss
# ═══════════════════════════════════════════════════════════════════════════════

class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):

        # 将 logits 映射到概率空间 [0, 1]
        pred = torch.sigmoid(pred)

        # 展平为 1D 向量以进行逐像素计算
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)

        # Dice 系数计算
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice


# ═══════════════════════════════════════════════════════════════════════════════
# 标准 BCE Loss 封装
# ═══════════════════════════════════════════════════════════════════════════════

class BCELoss(nn.Module):

    def __init__(self, pos_weight=2.0):
        super().__init__()
        self.pos_weight = torch.tensor([pos_weight])

    def forward(self, pred, target):
        """计算 BCE Loss。

        Args:
            pred  : 模型预测 logits (B, 1, H, W)。
            target: 真实标签 (B, 1, H, W)。

        Returns:
            scalar: 批次平均 BCE Loss。
        """
        # 将 pos_weight 移到与 pred 相同的设备
        if self.pos_weight.device != pred.device:
            self.pos_weight = self.pos_weight.to(pred.device)

        return F.binary_cross_entropy_with_logits(
            pred, target, pos_weight=self.pos_weight
        )


# ═══════════════════════════════════════════════════════════════════════════════
# BCEDiceLoss — 混合损失函数
# ═══════════════════════════════════════════════════════════════════════════════

class BCEDiceLoss(nn.Module):

    def __init__(self, w_bce=0.5, pos_weight=2.0, dice_smooth=1.0):
        super().__init__()
        if not 0.0 <= w_bce <= 1.0:
            raise ValueError(f"w_bce 必须在 [0, 1] 范围内, 当前值: {w_bce}")

        self.w_bce = w_bce
        self.w_dice = 1.0 - w_bce

        self.bce_loss = BCELoss(pos_weight=pos_weight)
        self.dice_loss = DiceLoss(smooth=dice_smooth)

    def forward(self, pred, target):
       
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)

        total = self.w_bce * bce + self.w_dice * dice

        return {
            'loss': total,
            'bce_loss': bce,
            'dice_loss': dice,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 损失函数工厂
# ═══════════════════════════════════════════════════════════════════════════════

def get_loss(loss_type='bce_dice', **kwargs):
   
    loss_type = loss_type.lower()

    if loss_type == 'bce':
        return BCELoss(
            pos_weight=kwargs.get('pos_weight', 2.0),
        )
    elif loss_type == 'dice':
        return DiceLoss(
            smooth=kwargs.get('dice_smooth', 1.0),
        )
    elif loss_type == 'bce_dice':
        return BCEDiceLoss(
            w_bce=kwargs.get('w_bce', 0.5),
            pos_weight=kwargs.get('pos_weight', 2.0),
            dice_smooth=kwargs.get('dice_smooth', 1.0),
        )
    else:
        raise ValueError(
            f"不支持的损失函数类型: {loss_type}。"
            f"可选: 'bce', 'dice', 'bce_dice'"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 模块自测代码
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("损失函数模块自测")
    print("=" * 70)

    # 模拟预测与标签 (B=2, C=1, H=16, W=16)
    torch.manual_seed(42)
    pred = torch.randn(2, 1, 16, 16)  # logits
    target = torch.randint(0, 2, (2, 1, 16, 16)).float()

    # 测试 1: BCE Loss
    print("\n[测试 1] BCE Loss")
    bce = get_loss('bce')
    bce_val = bce(pred, target)
    print(f"  BCE Loss: {bce_val.item():.4f}")

    # 测试 2: Dice Loss
    print("\n[测试 2] Dice Loss")
    dice = get_loss('dice')
    dice_val = dice(pred, target)
    print(f"  Dice Loss: {dice_val.item():.4f}")

    # 测试 3: BCEDiceLoss (w_bce=0.5)
    print("\n[测试 3] BCEDiceLoss (w_bce=0.5)")
    bce_dice = get_loss('bce_dice', w_bce=0.5)
    result = bce_dice(pred, target)
    print(f"  Total Loss : {result['loss'].item():.4f}")
    print(f"  BCE 分量   : {result['bce_loss'].item():.4f}")
    print(f"  Dice 分量  : {result['dice_loss'].item():.4f}")

    # 测试 4: 极端不平衡场景验证
    print("\n[测试 4] 极端不平衡场景 (仅 0.1% 像素为前景)")
    pred_unbalanced = torch.zeros(2, 1, 256, 1600)  # 全背景预测
    target_unbalanced = torch.zeros(2, 1, 256, 1600)
    # 仅设置 2 个像素为前景 (模拟 1:200000 的不平衡比)
    target_unbalanced[0, 0, 100, 100] = 1.0
    target_unbalanced[0, 0, 200, 200] = 1.0

    bce_val2 = bce(pred_unbalanced, target_unbalanced)
    dice_val2 = dice(pred_unbalanced, target_unbalanced)
    bce_dice_result = bce_dice(pred_unbalanced, target_unbalanced)

    print(f"  BCE Loss (全背景预测, 极不平衡):  {bce_val2.item():.6f}")
    print(f"  Dice Loss (全背景预测, 极不平衡): {dice_val2.item():.6f}")
    print(f"  BCEDice Loss:                       {bce_dice_result['loss'].item():.6f}")
    print("  → Dice Loss = 1.0 表明模型完全遗漏缺陷，梯度信号更强。")
    print("  → BCE Loss 接近 0 可能导致梯度消失。"  )

    print("\n所有损失函数测试完成!")
