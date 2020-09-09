import cv2
import numpy as np
import sys
sys.path.append("..")


PATH_mask = "../data/NICE1/NICE1/coco_int/train/iris_edge_mask/" #训练和测试
# PATH_mask = "../data/NICE1/NICE1/coco_int/test/iris_edge_mask/" #训练和测试

def get_contour(img):
    """获取连通域

    :param img: 输入图片
    :return: 最大连通域
    """
    # 灰度化, 二值化, 连通域分析
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

    img_contour, contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    return img_gray, contours[0]


def main():

    # 1.导入图片
    img_src = cv2.imread(PATH_mask+"0001010070001.png")

    # 2.获取连通域
    img_gray, contour = get_contour(img_src)

    # 3.轮廓外边缘点
    mask_out_edge = np.zeros(img_gray.shape, np.uint8)
    cv2.drawContours(mask_out_edge, [contour], 0, 255, 2)

    pixel_point1 = cv2.findNonZero(mask_out_edge)
    print("pixel point shape:", pixel_point1.shape)
    print("pixel point:\n", pixel_point1)

    # 4.轮廓内部
    mask_inside = np.zeros(img_gray.shape, np.uint8)
    cv2.drawContours(mask_inside, [contour], 0, 255, -1)
    pixel_point2 = cv2.findNonZero(mask_inside)

    print("pixel point2 shape:", pixel_point2.shape)
    print("pixel point2:\n", pixel_point2)

    # 5.显示图片
    cv2.imshow("img_src", img_src)
    cv2.imshow("mask_out_edge", mask_out_edge)
    cv2.imshow("mask_inside", mask_inside)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

