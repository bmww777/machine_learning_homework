

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# 基础卷积组件
# ═══════════════════════════════════════════════════════════════════════════════

class DoubleConv(nn.Module):


    def __init__(self, in_channels, out_channels, mid_channels=None):

        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UpsampleBlock(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels):
        """
        Args:
            in_channels  : 上一层 Decoder (或 Bottleneck) 输出的通道数。
            skip_channels: 对应 Encoder 层的通道数 (用于 Skip Connection)。
            out_channels : 本模块输出的通道数。
        """
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
 
        x = self.up(x)

        # 处理因下采样导致的尺寸奇偶不一致问题 (如从 13×13 上采样到 26×26)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode='bilinear', align_corners=True)

        x = torch.cat([x, skip], dim=1)  # 通道维度拼接
        x = self.conv(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# Encoder 工厂
# ═══════════════════════════════════════════════════════════════════════════════

class _ResNetEncoder(nn.Module):

    def __init__(self, backbone_name, pretrained=True):
 
        super().__init__()
        import torchvision.models as tv_models

        model_fn = getattr(tv_models, backbone_name, None)
        if model_fn is None:
            raise ValueError(
                f"不支持的 backbone: {backbone_name}。"
                f"可用的 torchvision ResNet: resnet18, resnet34, resnet50, resnet101"
            )

        backbone = model_fn(weights='IMAGENET1K_V1' if pretrained else None)

        # stem: conv1(7×7, stride=2) + bn + relu → 1/2 分辨率
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
        )
        # maxpool: 3×3, stride=2 → 1/4 分辨率
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1  # 1/4
        self.layer2 = backbone.layer2  # 1/8
        self.layer3 = backbone.layer3  # 1/16
        self.layer4 = backbone.layer4  # 1/32

        # ── 确定各层输出通道数 ──
        # ResNet18/34 使用 BasicBlock, 各层通道: [64, 64, 128, 256, 512]
        # ResNet50/101 使用 Bottleneck, 各层通道: [64, 256, 512, 1024, 2048]
        if '18' in backbone_name or '34' in backbone_name:
            self.channels = [64, 64, 128, 256, 512]
        else:
            self.channels = [64, 256, 512, 1024, 2048]

    def forward(self, x):
        features = []

        s0 = self.stem(x)            # 1/2 分辨率
        features.append(s0)

        x = self.maxpool(s0)          # 1/4
        x = self.layer1(x)
        features.append(x)            # 1/4

        x = self.layer2(x)
        features.append(x)            # 1/8

        x = self.layer3(x)
        features.append(x)            # 1/16

        x = self.layer4(x)
        features.append(x)            # 1/32

        return features


def _create_resnet_encoder(backbone_name, pretrained):
    """创建 ResNet Encoder 的便捷函数。

    Returns:
        (encoder_module, channels_list)
    """
    encoder = _ResNetEncoder(backbone_name, pretrained=pretrained)
    return encoder, encoder.channels


def _create_efficientnet_encoder(backbone_name, pretrained):

    try:
        import timm
    except ImportError:
        raise ImportError(
            "使用 EfficientNet 骨干网络需要安装 timm 库。\n"
            "请执行: pip install timm"
        )

    # timm 模型名直接使用用户传入的名称 (如 efficientnet_b0, efficientnet_b4)
    # 这些名称在 timm 0.9+ 中可直接识别
    timm_model_name = backbone_name

    encoder = timm.create_model(
        timm_model_name,
        pretrained=pretrained,
        features_only=True,
        out_indices=(0, 1, 2, 3, 4),  # 输出 5 个尺度的特征图
    )

    # 通过前向一次 dummy input 获取各层输出通道数
    with torch.no_grad():
        dummy = torch.randn(1, 3, 256, 256)
        features = encoder(dummy)
        channels = [f.shape[1] for f in features]

    return encoder, channels


def create_encoder(backbone_name, pretrained=True):

    backbone_name = backbone_name.lower()

    if backbone_name.startswith('resnet') or backbone_name.startswith('resnext'):
        return _create_resnet_encoder(backbone_name, pretrained)
    elif backbone_name.startswith('efficientnet'):
        return _create_efficientnet_encoder(backbone_name, pretrained)
    else:
        raise ValueError(
            f"不支持的 backbone: {backbone_name}。\n"
            f"支持的 ResNet 系列: resnet18, resnet34, resnet50, resnet101\n"
            f"支持的 EfficientNet 系列: efficientnet_b0, efficientnet_b1, ..., efficientnet_b4"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# UNetMultiBackbone — 主模型
# ═══════════════════════════════════════════════════════════════════════════════

class UNetMultiBackbone(nn.Module):


    def __init__(self, backbone_name='resnet34', pretrained=True,
                 in_channels=3, num_classes=1):
  
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        # ── Encoder ──
        self.encoder, self.enc_channels = create_encoder(
            backbone_name, pretrained=pretrained
        )
        # enc_channels: [ch_skip0, ch_skip1, ch_skip2, ch_skip3, ch_bottleneck]

        ch_skip0, ch_skip1, ch_skip2, ch_skip3, ch_bottleneck = self.enc_channels

        # ── Decoder (4 个上采样块) ──
        # 每层输出通道 = 对应跳跃连接层的通道数
        self.dec0 = UpsampleBlock(
            in_channels=ch_bottleneck,
            skip_channels=ch_skip3,
            out_channels=ch_skip3,
        )
        self.dec1 = UpsampleBlock(
            in_channels=ch_skip3,
            skip_channels=ch_skip2,
            out_channels=ch_skip2,
        )
        self.dec2 = UpsampleBlock(
            in_channels=ch_skip2,
            skip_channels=ch_skip1,
            out_channels=ch_skip1,
        )
        self.dec3 = UpsampleBlock(
            in_channels=ch_skip1,
            skip_channels=ch_skip0,
            out_channels=ch_skip0,
        )

        # ── 分割头 ──
        # 将最高分辨率特征图上采样到原图尺寸
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 轻量分割头: 1×1 卷积将特征映射为类别分数
        self.seg_head = nn.Sequential(
            nn.Conv2d(ch_skip0, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x):

        # ── Encoder: 提取 5 个尺度的特征 ──
        features = self.encoder(x)
        skip0, skip1, skip2, skip3, bottleneck = features

        # ── Decoder: 逐级上采样 + 跳跃连接 ──
        d0 = self.dec0(bottleneck, skip3)
        d1 = self.dec1(d0, skip2)
        d2 = self.dec2(d1, skip1)
        d3 = self.dec3(d2, skip0)

        # ── 分割头 ──
        out = self.final_up(d3)

        # 处理因下采样取整导致的尺寸不匹配 (理论上 256×1600 不存在此问题)
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:],
                                mode='bilinear', align_corners=True)

        logits = self.seg_head(out)
        return logits

    def get_encoder_channels(self):
        """返回 Encoder 各层的通道数，用于外部调试或自定义 Decoder。"""
        return self.enc_channels


# ═══════════════════════════════════════════════════════════════════════════════
# 模块自测代码
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("UNetMultiBackbone 模型自测")
    print("=" * 70)

    # 测试 1: ResNet34 骨干
    print("\n[测试 1] 骨干网络: resnet34")
    model = UNetMultiBackbone(backbone_name='resnet34', pretrained=False, num_classes=1)
    dummy_input = torch.randn(2, 3, 256, 1600)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"  输入形状: {dummy_input.shape}")
    print(f"  输出形状: {output.shape}")
    print(f"  Encoder 各层通道数: {model.get_encoder_channels()}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数量: {total_params / 1e6:.2f}M")
    assert output.shape == (2, 1, 256, 1600), "输出形状错误!"
    print("  ✓ ResNet34 测试通过")

    # 测试 2: ResNet50 骨干
    print("\n[测试 2] 骨干网络: resnet50")
    model2 = UNetMultiBackbone(backbone_name='resnet50', pretrained=False, num_classes=1)
    with torch.no_grad():
        output2 = model2(dummy_input)
    print(f"  输出形状: {output2.shape}")
    print(f"  Encoder 各层通道数: {model2.get_encoder_channels()}")
    total_params2 = sum(p.numel() for p in model2.parameters())
    print(f"  总参数量: {total_params2 / 1e6:.2f}M")
    assert output2.shape == (2, 1, 256, 1600)
    print("  ✓ ResNet50 测试通过")

    print("\n所有模型自测完成!")
