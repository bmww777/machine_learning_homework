
import numpy as np

import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ═══════════════════════════════════════════════════════════════════════════════
# RLE 解码函数
# ═══════════════════════════════════════════════════════════════════════════════

def rle2mask(rle_str, shape=(256, 1600)):
    """将 RLE (Run-Length Encoding) 行程编码字符串解码为二维二值 Mask。

    算法步骤:
      1. 解析 RLE 字符串，提取起始位置 (starts) 与行程长度 (lengths)。
      2. 将 1-based 索引转换为 0-based。
      3. 创建一维扁平数组，将对应区间的像素置 1。
      4. 以列优先 (Fortran order='F') 方式 reshape 为 (H, W) 的二维 Mask。

    Args:
        rle_str: RLE 编码字符串，如 "29102 12 29346 24 29602 24"。
                 若为 NaN、空字符串或 "-1" 则视为无缺陷，返回全零 Mask。
        shape : (height, width) 元组，默认 (256, 1600)。

    Returns:
        np.ndarray: 二值 Mask，形状 (height, width)，dtype=np.uint8，取值 {0, 1}。
    """
    # ── 处理无缺陷样本 ──
    if isinstance(rle_str, float) and np.isnan(rle_str):
        return np.zeros(shape, dtype=np.uint8)
    if rle_str is None or str(rle_str).strip() in ('', '-1', 'nan'):
        return np.zeros(shape, dtype=np.uint8)

    rle_str = str(rle_str).strip()
    tokens = rle_str.split()

    # 奇数位置为起始像素 (1-based)，偶数位置为行程长度
    starts = np.asarray(tokens[0::2], dtype=np.int64)
    lengths = np.asarray(tokens[1::2], dtype=np.int64)

    # 转换为 0-based 索引
    starts -= 1
    ends = starts + lengths

    # 创建一维数组并填充
    total_pixels = shape[0] * shape[1]
    flat_mask = np.zeros(total_pixels, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        flat_mask[lo:hi] = 1

    # 列优先 reshape 以匹配 RLE 的 Fortran 排列顺序
    mask = flat_mask.reshape(shape, order='F')
    return mask


# ═══════════════════════════════════════════════════════════════════════════════
# 数据增强流水线 (Albumentations)
# ═══════════════════════════════════════════════════════════════════════════════

def get_train_transforms(image_size=(256, 1600)):
    """返回训练集的数据增强 Compose 流水线。

    训练增强策略:
      - 水平翻转 (50% 概率): 钢铁表面缺陷方向无关，翻转不改变语义。
      - 随机亮度/对比度调整: 模拟不同光照条件下的产线成像。
      - 随机 Gamma 校正: 增强对不同曝光水平的鲁棒性。
      - 归一化: ImageNet 均值与标准差，迁移 ImageNet 预训练权重。
      - ToTensorV2: 将 HWC uint8 图像转为 CHW float32 Tensor。

    Args:
        image_size: (H, W) 元组，需与数据集图像尺寸一致。

    Returns:
        albumentations.Compose 对象。
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.2,
                                   p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),   # ImageNet 均值
            std=(0.229, 0.224, 0.225),    # ImageNet 标准差
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])


def get_val_transforms(image_size=(256, 1600)):
    """返回验证集的数据预处理 Compose 流水线（无数据增强，仅归一化）。"""
    return A.Compose([
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# PyTorch Dataset 类
# ═══════════════════════════════════════════════════════════════════════════════

class SteelDataset(Dataset):
    """Severstal 钢铁缺陷语义分割数据集。

    从 Kaggle CSV 格式的标注文件中加载图像与对应的 RLE Mask，
    合并多类别缺陷为二值语义分割标签 (0=背景, 1=缺陷)。

    数据格式要求:
      - CSV 文件至少包含两列: "ImageId" (图像文件名不含后缀) 和 "EncodedPixels" (RLE 字符串)。
      - 图像文件存放在 image_dir 目录下，文件名为 "{ImageId}.jpg"。
      - 若一张图像包含多条记录 (多个 ClassId)，则合并所有 RLE Mask 为一张二值 Mask。

    Usage:
        df = pd.read_csv("train.csv")
        dataset = SteelDataset(df, image_dir="train_images",
                               transforms=get_train_transforms())
        image, mask, image_id = dataset[0]
    """

    def __init__(self, df, image_dir, transforms=None):
        """
        Args:
            df           : pandas DataFrame，必须包含 "ImageId" 和 "EncodedPixels" 列。
            image_dir    : str，存放训练图像的文件夹路径。
            transforms   : albumentations.Compose 对象或 None。
        """
        import pandas as pd
        self.df = df.copy()
        self.image_dir = image_dir
        self.transforms = transforms

        # 将 EncodedPixels 列中的 NaN 填充为 '' 以简化后续判断逻辑
        self.df['EncodedPixels'] = self.df['EncodedPixels'].fillna('')

        # 获取去重后的图像 ID 列表
        self.image_ids = self.df['ImageId'].unique().tolist()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # ── 1. 加载图像 (尝试常见扩展名) ──
        import cv2
        image = None
        for ext in ['', '.jpg', '.jpeg', '.png', '.bmp']:
            image_path = f"{self.image_dir}/{image_id}{ext}"
            candidate = cv2.imread(image_path)
            if candidate is not None:
                image = candidate
                break
        if image is None:
            raise FileNotFoundError(
                f"无法读取图像: {self.image_dir}/{image_id}[.jpg/.png/...] "
                f"— 请检查图像文件夹路径与文件扩展名"
            )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ── 2. 构建二值 Mask ──
        # 获取该图像在 CSV 中的所有记录 (可能对应多个缺陷类别)
        records = self.df[self.df['ImageId'] == image_id]
        mask = np.zeros((256, 1600), dtype=np.uint8)

        for _, row in records.iterrows():
            rle = row['EncodedPixels']
            if rle == '':
                continue  # 该类别无缺陷
            class_mask = rle2mask(rle, shape=(256, 1600))
            # 使用逐像素取最大值的方式合并多类别 Mask
            mask = np.maximum(mask, class_mask)

        mask = mask.astype(np.float32)

        # ── 3. 应用 Albumentations 数据增强 ──
        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # 若无 transforms，手动转为 Tensor 并归一化到 [0, 1]
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0)

        # mask 增加通道维度: (1, H, W)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        return image, mask, image_id


# ═══════════════════════════════════════════════════════════════════════════════
# 模块自测代码
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 测试 RLE 解码
    test_rle = "1 10 50 5"
    mask = rle2mask(test_rle, shape=(10, 10))
    print(f"RLE 解码测试 - Mask 形状: {mask.shape}, 前景像素数: {mask.sum()}")
    print("rle2mask 函数测试通过 ✓")
