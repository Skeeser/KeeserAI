import cv2
import numpy as np
import os
# 假设有一个包含图像文件名的列表 image_files，你可以遍历它并计算均值和标准差
mean = [0, 0, 0]
std = [0, 0, 0]

image_files = []
data_dir = "../../resource/yolo"

for i in range(50655):
    pic_path = os.path.join(data_dir, "PngImages", str(i) + ".png")
    image_files.append(pic_path)

for image_file in image_files:
    image = cv2.imread(image_file)
    image = image.astype(np.float32) / 255.0  # 将像素值转换为 [0, 1] 范围
    mean += np.mean(image, axis=(0, 1))
    std += np.std(image, axis=(0, 1))

mean /= len(image_files)
std /= len(image_files)

print(mean)
print(std)
