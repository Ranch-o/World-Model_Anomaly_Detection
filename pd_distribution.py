import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def get_image_pixel_values(image_path):
    """ 读取图像并返回其像素值 """
    with Image.open(image_path) as img:
        img_array = np.array(img)
        # 归一化像素值到0-1
        normalized_img_array = img_array / 255.0
        return normalized_img_array.flatten()

# 图像文件夹路径
folder_path = '/disk/vanishing_data/du541/anomaly_scores/scores'

# 获取所有图像文件名
image_files = sorted(os.listdir(folder_path), key=lambda x: int(x.split('_')[-1].split('.')[0]))

# 初始化直方图的bins和值
# 由于我们现在处理的是归一化的值，bins也应该在0-1之间
hist_bins = np.linspace(0, 1, 257)  # 包含0到1的257个点
hist_values = np.zeros(len(hist_bins) - 1, dtype=int)  # 减1因为边界比bins多一个

# 分批遍历图像文件并更新直方图
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    pixels = get_image_pixel_values(image_path)
    hist, _ = np.histogram(pixels, bins=hist_bins)
    hist_values += hist

# 绘制分布图，确保每个bin的宽度是1/255
plt.bar(hist_bins[:-1], hist_values, width=1/255, color='blue', alpha=0.7)
plt.title('Normalized Pixel Value Distribution')
plt.xlabel('Normalized Pixel Intensity')
plt.ylabel('Frequency')

# 保存图形
output_folder = '/disk/vanishing_data/du541/thesis_plot'  # 替换为您想保存图形的文件夹路径
output_filename = 'normalized_pixel_distribution.png'  # 您想保存的文件名
plt.savefig(os.path.join(output_folder, output_filename))

# 显示图形
plt.show()

