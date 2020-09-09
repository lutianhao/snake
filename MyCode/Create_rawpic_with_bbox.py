import sys
import os
sys.path.append("..")
import matplotlib.pyplot as plt
import imutils
import configparser
import numpy as np
import cv2


#1.先读入图片数据
#2.根据图片数据椭圆位置找到外切矩形坐标
#3.将矩形坐标等信息保存在新建的文件夹里面，可以是数值也可以是图片


Pic_PATH = '../data/NICE1/NICE1/coco_int/'
# Param_iris_bbox_PATH = '../data/NICE1/NICE1/coco_int/train/iris_edge_bbox/' #修改一下train、val和test
# Param_pupil_bbox_PATH = '../data/NICE1/NICE1/coco_int/train/pupil_edge_bbox/'
# Pic_Save_PATH = '../data/NICE1/NICE1/coco_int/train/'
Param_iris_bbox_PATH = '../data/NICE1/NICE1/coco_int/val/iris_edge_bbox/'
Param_pupil_bbox_PATH = '../data/NICE1/NICE1/coco_int/val/pupil_edge_bbox/'
Pic_Save_PATH = '../data/NICE1/NICE1/coco_int/val/'
# Param_iris_bbox_PATH = '../data/NICE1/NICE1/coco_int/test/iris_edge_bbox/'
# Param_pupil_bbox_PATH = '../data/NICE1/NICE1/coco_int/test/pupil_edge_bbox/'
# Pic_Save_PATH = '../data/NICE1/NICE1/coco_int/test/'





def getFileList(dir, Filelist, ext=None):
    """
    获取文件夹及其子文件夹中文件列表
    输入 dir：文件夹根目录
    输入 ext: 扩展名
    返回： 文件路径列表
    """
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-4:]:   #我们的拓展名是几个字符就改成几，如png 就是3；JPEG就是4
                Filelist.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getFileList(newDir, Filelist, ext)

    return Filelist


# org_img_folder = os.path.join(Pic_PATH,'train/image')    #这里修改train、val和test
org_img_folder = os.path.join(Pic_PATH,'val/image')
# org_img_folder = os.path.join(Pic_PATH,'test/image')

# 检索文件
print("当前图片路径为：{}".format(org_img_folder))
imglist = getFileList(org_img_folder, [], 'JPEG')
print('本次执行检索到 ' + str(len(imglist)) + ' 张图像\n')
i = 0
for imgpath in imglist:
    i+=1
    print("完成第{}/{}次".format(i, len(imglist)))
    imgname = os.path.splitext(os.path.basename(imgpath))[0]
    img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    #读取ini文件中心点长宽信息并在图上画出框
    #  实例化configParser对象
    config = configparser.ConfigParser()
    # -read读取ini文件
    if not os.path.isfile(Param_iris_bbox_PATH+imgname+'.ini'):
        print(Param_iris_bbox_PATH+imgname+'.ini')
        print("{}.ini 文件不存在".format(imgname))
    elif not os.path.isfile(Param_pupil_bbox_PATH+imgname+'.ini'):
        print(Param_pupil_bbox_PATH + imgname + '.ini')
        print("{}.ini 文件不存在".format(imgname))
    else:
        #画外框
        config.read(Param_iris_bbox_PATH + imgname + '.ini')
        iris_centerX = config.get('bbox', 'x')
        iris_centerY = config.get('bbox', 'y')
        iris_wdith = config.get('bbox', 'width')
        iris_height = config.get('bbox','height')
        x = int(iris_centerX)
        y = int(iris_centerY)
        w = int(iris_wdith)
        h = int(iris_height)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #画内框
        config.read(Param_pupil_bbox_PATH + imgname + '.ini')
        iris_centerX = config.get('bbox', 'x')
        iris_centerY = config.get('bbox', 'y')
        iris_wdith = config.get('bbox', 'width')
        iris_height = config.get('bbox', 'height')
        x = int(iris_centerX)
        y = int(iris_centerY)
        w = int(iris_wdith)
        h = int(iris_height)
        # 外接矩形框，没有方向角
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        rgb_tmp_img = cv2.merge(cv2.split(img))
        #输出图片
        cv2.imwrite(Pic_Save_PATH+'img_label/'+imgname+'.JPEG',rgb_tmp_img)
        #展示图片
        # plt.imshow(rgb_tmp_img)
        # plt.axis('off')
        # plt.show()

    # cv2.imshow('a', tmp_img)
    # cv2.imwrite('./result.jpg', tmp_img)
    # cv2.waitKey(0)

    #这段是用来查看输出白色区域像素点坐标的
# print('imgval:{}\n imgshape:{}\n imgtype:{}'.format(tmp_img,tmp_img.shape,tmp_img.dtype))
# for j in range(tmp_img.shape[1]):
#     for i in range(tmp_img.shape[0]):
#         # print(tmp_img[i,j,0])
#         if tmp_img[i,j,0] == 255:
#             print('坐标为：（{},{}）'.format(i,j))

    # 对每幅图像执行相关操作
    # img = plt.imread('002.jpg')
    # 图片的高H为460，宽W为346，颜色通道C为3
    # print(img.shape)
    # print(img.dtype)
    # print(type(img))
    # plt.imshow(img)
    # plt.axis('off')
    # # plt.savefig("/home/tianhao.lu/code/Deep_snake/snake/Result/tmp/{}.jpg".format(imgname) )
    # plt.show()