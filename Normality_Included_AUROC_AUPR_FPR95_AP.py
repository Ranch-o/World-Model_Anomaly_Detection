import os
import random
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve, average_precision_score

# 设定文件路径
anomaly_scores_dir = '/disk/vanishing_data/du541/anomaly_scores/scores'
binary_masks_dir = '/disk/vanishing_data/du541/binary_masks'

# 检查掩码图像以确定异常
def contains_anomaly(mask_image):
    return np.any(mask_image == 255)  # 假设异常像素值为255

# 获取文件列表
anomaly_images = []  # 填充含异常值的图片路径
normal_images = []   # 填充不含异常值的图片路径

# 遍历掩码目录中的所有文件
for filename in os.listdir(binary_masks_dir):
    if filename.endswith(".png"):  # 假设所有掩码文件都是PNG格式
        mask_path = os.path.join(binary_masks_dir, filename)
        mask_image = cv2.imread(mask_path, 0)  # 以灰度模式读取
        
        # 根据掩码图像判断图片类别
        if contains_anomaly(mask_image):
            anomaly_images.append(filename.replace('binary_mask_', 'anomaly_scores_frame_'))
        else:
            normal_images.append(filename.replace('binary_mask_', 'anomaly_scores_frame_'))

# 分配不含异常值的图片到含异常值的图片
batch_size = 5  # 每批处理的图片数
batches = []

for anomaly_image in anomaly_images:
    batch = [anomaly_image]
    batch.extend(random.sample(normal_images, batch_size - 1))
    batches.append(batch)

# 初始化用于聚合的变量
aggregate_scores = []
aggregate_masks = []

# 处理每个批次
for batch in batches:
    scores_flat = []
    masks_flat = []

    for image_path in batch:
        anomaly_score_path = os.path.join(anomaly_scores_dir, image_path)
        binary_mask_path = os.path.join(binary_masks_dir, image_path.replace('anomaly_scores_frame_', 'binary_mask_'))

        anomaly_score_img = cv2.imread(anomaly_score_path, 0) / 255.0
        binary_mask = cv2.imread(binary_mask_path, 0) / 255.0

        scores_flat.extend(anomaly_score_img.flatten())
        masks_flat.extend(binary_mask.flatten())

    # 聚合批次评估结果
    aggregate_scores.extend(scores_flat)
    aggregate_masks.extend(masks_flat)

# 将列表转换为NumPy数组
aggregate_scores = np.array(aggregate_scores)
aggregate_masks = np.array(aggregate_masks)

# 计算整体评估指标
auroc = roc_auc_score(aggregate_masks, aggregate_scores)
precision, recall, _ = precision_recall_curve(aggregate_masks, aggregate_scores)
aupr = auc(recall, precision)
fpr, tpr, _ = roc_curve(aggregate_masks, aggregate_scores)
fpr95 = fpr[np.where(tpr >= 0.95)[0][0]]
ap = average_precision_score(aggregate_masks, aggregate_scores)

# 输出结果
print(f"Total AUROC: {auroc}")
print(f"Total AUPR: {aupr}")
print(f"Total FPR95: {fpr95}")
print(f"Total AP: {ap}")