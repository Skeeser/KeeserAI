import os
import sys
import re
import shutil
import cv2
from xml.etree.ElementTree import *

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
#

# 此文件数据评估
Annotation_Path = "../../resource/yolo/annotations/"
Target_Path = "../../resource/yolo/tricked_annotations"
Pic_Path = "../../resource/yolo/PngImages"
Target_ids = "../resource/yolo/val_annotation_ids.txt"
path_all_num = 0
cur_path_num = 0


class VOCAnnotationTransform(object):
    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                # cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            # 默认就是math
            label_idx = 0  #  self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


with open(Target_ids, 'w') as ids_file:
    # 遍历文件夹
    # 使用 os.walk() 方法遍历文件夹
    for root, dirs, files in os.walk(Annotation_Path):
        if path_all_num == 0:
            path_all_num = len(dirs)


        # 遍历当前文件夹中的所有子文件夹
        # for directory in dirs:
        #     dir_num = int(directory)
        #     if dir_num % 2 == 0:
        #         print(f"子文件夹：{os.path.join(root, directory)}")
        if len(files) > 0 and files[0].split('.')[1] == "xml":
            # 在xml文件夹下
            # print(f"当前文件夹：{root}")
            match = re.search(r'annotations/(\d+)', root)

            if match:
                extracted_number = match.group(1)
                cur_path_num += 1
                # print(extracted_number)
                if int(extracted_number) % 2 == 1:
                    # 奇数
                    print(extracted_number)
                    file_path = os.path.join(root, files[0])
                    picture_path = os.path.join(Pic_Path, files[0].split('.')[0] + ".png")
                    # cv2.imwrite('-1.jpg', img)
                    img = cv2.imread(picture_path)
                    target = ET.parse(file_path).getroot()
                    target_transform = VOCAnnotationTransform()
                    if target_transform is not None:
                        target = target_transform(target)
                    for box in target:
                        xmin, ymin, xmax, ymax, _ = box
                        img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
                    cv2.imshow(f'user:{extracted_number}', img)
                    key = cv2.waitKey(0)
                    if key == ord('q'):  # 如果按下 'q' 键，则退出循环
                        continue
                    elif key == ord('s'):  # 如果按下 's' 键，则保存图像
                        print('extracted_number saved!')
                        ids_file.write(extracted_number + '\n')

            else:
                print("无法识别文件夹", root)
            # print(files[0])


print("总的文件夹数:", path_all_num)
print("遍历成功的文件夹数", cur_path_num)