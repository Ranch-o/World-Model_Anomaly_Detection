import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def get_image_pixel_values(image_path):
    """ 读取图像并返回其RGB三通道的像素值 """
    with Image.open(image_path) as img:
        img = np.array(img)
        return img[:, :, 0].flatten(), img[:, :, 1].flatten(), img[:, :, 2].flatten()

# 图像文件夹路径
folder_path = '/disk/vanishing_data/du541/anomaly_scores/perceptual_difference'

# 获取所有图像文件名
image_files = sorted(os.listdir(folder_path), key=lambda x: int(x.split('_')[-1].split('.')[0]))

# 初始化直方图的bins和值
hist_bins = np.linspace(0, 256, 257)  # 包含256个边界的257个点
hist_values_r = np.zeros(256, dtype=int)
hist_values_g = np.zeros(256, dtype=int)
hist_values_b = np.zeros(256, dtype=int)

# 分批遍历图像文件并更新直方图
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    pixels_r, pixels_g, pixels_b = get_image_pixel_values(image_path)
    hist_r, _ = np.histogram(pixels_r, bins=hist_bins)
    hist_g, _ = np.histogram(pixels_g, bins=hist_bins)
    hist_b, _ = np.histogram(pixels_b, bins=hist_bins)
    hist_values_r += hist_r
    hist_values_g += hist_g
    hist_values_b += hist_b

# 绘制分布图
plt.figure(figsize=(10, 6))
plt.bar(hist_bins[:-1] - 0.5, hist_values_r, width=1, color='red', alpha=0.7, label='Red Channel')
plt.bar(hist_bins[:-1] - 0.5, hist_values_g, width=1, color='green', alpha=0.7, label='Green Channel')
plt.bar(hist_bins[:-1] - 0.5, hist_values_b, width=1, color='blue', alpha=0.7, label='Blue Channel')
plt.title('Pixel Value Distribution by Channel')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.legend()


# 保存图形
output_folder = '/disk/vanishing_data/du541/thesis_plot'  # 替换为您想保存图形的文件夹路径
output_filename = 'pixel_distribution_RGB.png'  # 您想保存的文件名
plt.savefig(os.path.join(output_folder, output_filename))

# 显示图形
plt.show()


