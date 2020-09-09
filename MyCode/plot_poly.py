# 这里绘制polygen的图，看看作为边缘点的polygen是否跟想象的一致

import matplotlib.pyplot as plt
import json
import numpy as np
import os
import sys
import cv2
import configparser

output=[]
list = [[173, 92], [172, 93], [169, 93], [168, 94], [165, 94], [164, 95], [163, 95], [162, 96], [161, 96], [160, 97], [159, 97], [157, 99], [156, 99], [153, 102], [152, 102], [149, 105], [149, 106], [146, 109], [146, 110], [145, 111], [145, 112], [144, 113], [144, 114], [143, 115], [143, 116], [142, 117], [142, 119], [141, 120], [141, 124], [140, 125], [140, 135], [141, 136], [141, 140], [142, 141], [142, 143], [143, 144], [143, 145], [144, 146], [144, 147], [145, 148], [145, 149], [147, 151], [147, 152], [155, 160], [156, 160], [158, 162], [159, 162], [160, 163], [161, 163], [162, 164], [163, 164], [164, 165], [165, 165], [166, 166], [169, 166], [170, 167], [175, 167], [176, 168], [183, 168], [184, 167], [189, 167], [190, 166], [193, 166], [194, 165], [195, 165], [196, 164], [197, 164], [198, 163], [199, 163], [200, 162], [201, 162], [203, 160], [204, 160], [212, 152], [212, 151], [214, 149], [214, 148], [215, 147], [215, 146], [216, 145], [216, 144], [217, 143], [217, 141], [218, 140], [218, 136], [219, 135], [219, 125], [218, 124], [218, 120], [217, 119], [217, 117], [216, 116], [216, 115], [215, 114], [215, 113], [214, 112], [214, 111], [213, 110], [213, 109], [210, 106], [210, 105], [207, 102], [206, 102], [203, 99], [202, 99], [200, 97], [199, 97], [198, 96], [197, 96], [196, 95], [195, 95], [194, 94], [191, 94], [190, 93], [187, 93], [186, 92]]
print(list[3],len(list[3]))
for tmp in list:
    output.append(tmp[0])
    output.append(tmp[1])
print(output)

# print(len(list))
# print(list[1])
# print(list[1][0])
# print(list[1][1])
# list_x = [tmplist[0] for tmplist in list]
# list_y = [tmplist[1] for tmplist in list]
# plt.plot(list_x,list_y)
# plt.show()


#这里做一下单张图片的轮廓检索，看下是findcontours出的问题，还是findnozero出的问题


# import cv2
# import numpy as np
#
#
# def get_contour(img):
#     """获取连通域
#
#     :param img: 输入图片
#     :return: 最大连通域
#     """
#     # 灰度化, 二值化, 连通域分析
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     ret, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
#
#     img_contour, contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#
#     return img_gray, contours[0]
#
#
# def main():
#
#     # 1.导入图片
#     img_src = cv2.imread("../data/NICE1/NICE1/coco_int/test/iris_edge_mask/0002010140001.png")
#
#     # 2.获取连通域
#     img_gray, contour = get_contour(img_src)
#     print("contour: {}".format(contour.shape))
#     # 3.轮廓外边缘点
#     mask_out_edge = np.zeros(img_gray.shape, np.uint8)
#     cv2.drawContours(mask_out_edge, [contour], 0, 255, 2)
#
#     pixel_point1 = cv2.findNonZero(mask_out_edge)
#     print("pixel point shape:", pixel_point1.shape)
#     print("pixel point:\n", pixel_point1)
#
#     # 4.轮廓内部
#     mask_inside = np.zeros(img_gray.shape, np.uint8)
#     cv2.drawContours(mask_inside, [contour], 0, 255, -1)
#     pixel_point2 = cv2.findNonZero(mask_inside)
#
#     print("pixel point2 shape:", pixel_point2.shape)
#     print("pixel point2:\n", pixel_point2)
#
#     # 5.显示图片
#     plt.imshow("img_src", img_src)
#     plt.imshow("mask_out_edge", mask_out_edge)
#     plt.imshow("mask_inside", mask_inside)
#
#     plt.show()
#     # x = contour[:,0,0]  #contour获取的是按顺序的
#     # y = contour[:,0,1]
#     # x = pixel_point1[:,0,0]#pixel point1获取的是一行一行扫描的
#     # y = pixel_point1[:,0,1]
#     x = mask_out_edge[:,0,0]
#     y = mask_out_edge[:,0,1]
#     print(mask_out_edge)
#     print(mask_out_edge.shape)
#     # plt.plot(x,y)
#     # plt.show()
#
# if __name__ == '__main__':
#     main()




