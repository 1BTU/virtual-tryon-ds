import os
import random

# # 指定待读取的文件夹路径
# folder_path = 'VITON_train/clothes'
#
# # 获取文件夹中所有图片的文件名
# file_names = [file for file in os.listdir(folder_path) if file.endswith('.jpg') or file.endswith('.png')]
#
# # 随机打乱文件名列表
# random.shuffle(file_names)
#
# # 将文件名列表写入txt文件
# with open('cloth_random.txt', 'w') as f:
#     for file_name in file_names:
#         f.write(file_name + '\n')

# 合并两个txt文件的内容到一个新的txt文件
with open('train_unpairs.txt.txt', 'w') as f:
    with open('train.txt', 'r') as f1:
        with open('cloth_random.txt', 'r') as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()
            max_lines = max(len(lines1), len(lines2))

            for i in range(max_lines):
                if i < len(lines1):
                    f.write(lines1[i].strip('\n') + '\t')
                else:
                    f.write('\t')

                if i < len(lines2):
                    f.write(lines2[i].strip('\n'))

                f.write('\n')