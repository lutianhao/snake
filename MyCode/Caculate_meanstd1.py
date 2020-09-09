# 首先先把数据集的图片路径保存在一个txt文件夹里面

import os
import sys
sys.path.append("..")

def generate(dir, label):
    listText = open('list.txt', 'a')
    for file in dir:
        fileType = os.path.split(file)
        if fileType[1] == '.txt':
            continue
        name = file + ' ' + str(int(label)) + '\n'
        listText.write(name)
    listText.close()


outer_path = '../data/NICE1/NICE1/coco_int/train/image/'  # 这里是你的图片的目录
# outer_path = '../data/NICE1/NICE1/coco_int/test/image/'  # 这里是你的图片的目录

if __name__ == '__main__':
    i = 1
    num = 0
    fingerlist = os.listdir(outer_path)
    fingerlist.sort()

    for finger in fingerlist:
        finallPATH = os.path.join(outer_path, finger)
        finallPATH = finallPATH.replace('\\', '/')

        listText = open(outer_path+'image_list.txt', 'a')
        fileType = os.path.split(finallPATH)

        name = finallPATH + '\n'

        listText.write(name)

    listText.close()
    i += 1