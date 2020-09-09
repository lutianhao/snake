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


Pic_PATH = '../data/NICE1/NICE1/coco_int'
# Param_Save_PATH = '../data/NICE1/NICE1/coco_int/test/iris_edge_bbox/' #这里一个是iris一个是pupil，注意：要更改test、train、val
Param_Save_PATH = '../data/NICE1/NICE1/coco_int/train/pupil_edge_bbox/'

import os
import cv2


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
            if ext in dir[-3:]:
                Filelist.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getFileList(newDir, Filelist, ext)

    return Filelist


org_img_folder = os.path.join(Pic_PATH,'train/pupil_edge_mask')   #这里也要换一下！

# 检索文件
imglist = getFileList(org_img_folder, [], 'png')
print('本次执行检索到 ' + str(len(imglist)) + ' 张图像\n')
i = 0
for imgpath in imglist:
    i+=1
    print("完成第{}/{}次".format(i, len(imglist)))
    imgname = os.path.splitext(os.path.basename(imgpath))[0]
    img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    tmp_img = img.copy()
    imgray = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 大津阈值
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # cv2.RETR_EXTERNAL 定义只检测外围轮廓

    cnts = contours[0] if imutils.is_cv2() else contours[1]  # 用imutils来判断是opencv是2还是2+

    for cnt in cnts:
        # 外接矩形框，没有方向角
        x, y, w, h = cv2.boundingRect(cnt)
        #保存bbox相关信息到ini文件中去
        tmp_inipath = os.path.join(Param_Save_PATH, "{}.ini".format(imgname))
        if os.path.exists(tmp_inipath):  #如果已存在ini文件，则删除了重新生成
            os.remove(tmp_inipath)
        config = configparser.ConfigParser()
        config['bbox'] = {
            'x':x,
            'y':y,
            'width':w,
            'height':h
        }
        with open(tmp_inipath,'w') as cfg:
            config.write(cfg)
        cv2.rectangle(tmp_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # # 最小外接矩形框，有方向角
        # rect = cv2.minAreaRect(cnt)
        # box = cv2.cv.Boxpoints() if imutils.is_cv2() else cv2.boxPoints(rect)
        # box = np.int0(box)
        # cv2.drawContours(tmp_img, [box], 0, (0, 0, 255), 2)

        # # 最小外接圆
        # (x, y), radius = cv2.minEnclosingCircle(cnt)
        # center = (int(x), int(y))
        # radius = int(radius)
        # cv2.circle(tmp_img, center, radius, (255, 0, 0), 2)

        # # 椭圆拟合
        # ellipse = cv2.fitEllipse(cnt)
        # cv2.ellipse(tmp_img, ellipse, (255, 255, 0), 2)

        # # 直线拟合
        # rows, cols = tmp_img.shape[:2]
        # [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
        # lefty = int((-x * vy / vx) + y)
        # righty = int(((cols - x) * vy / vx) + y)
        # tmp_img = cv2.line(tmp_img, (cols - 1, righty), (0, lefty), (0, 255, 255), 2)
    rgb_tmp_img = cv2.merge(cv2.split(tmp_img))
    plt.imshow(rgb_tmp_img)
    plt.axis('off')
    plt.show()

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