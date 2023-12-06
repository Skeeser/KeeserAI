import os
import re
import shutil

# 此文件数据预处理
Annotation_Path = "../resource/yolo/annotations/"
Target_Path = "../resource/yolo/tricked_annotations"
Target_ids = "../resource/yolo/yolo_ids.txt"
path_all_num = 0
cur_path_num = 0

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
                    for file in files:
                        if file.split('.')[1] == "xml":
                            file_path = os.path.join(root, file)
                            ids_file.write(file.split('.')[0] + '\n')
                            # print(file_path)
                            shutil.copy(file_path, Target_Path)


            else:
                print("无法识别文件夹", root)
            # print(files[0])


print("总的文件夹数:", path_all_num)
print("遍历成功的文件夹数", cur_path_num)