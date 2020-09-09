import json
import numpy as np
import os
import sys
import cv2
import configparser


if __name__ == '__main__':                            #设置相关路径
    Pic_PATH = '../data/NICE1/NICE1/' #图片路径，注意修改下面的train 或者test路径
    Param_bbox_PATH = '../data/NICE1/NICE1/coco_int/' #bbox路径
    # Json_save_PATH = '../data/NICE1/NICE1/coco_int/train/annotations/NICE_train.json' #json输出路径
    # iris_Mask_PATH = '../data/NICE1/NICE1/coco_int/train/iris_edge_mask/'    #mask文件路径
    # pupil_Mask_PATH = '../data/NICE1/NICE1/coco_int/train/pupil_edge_mask/'
    # Json_save_PATH = '../data/NICE1/NICE1/coco_int/val/annotations/NICE_val.json' #json输出路径
    # iris_Mask_PATH = '../data/NICE1/NICE1/coco_int/val/iris_edge_mask/'    #mask文件路径
    # pupil_Mask_PATH = '../data/NICE1/NICE1/coco_int/val/pupil_edge_mask/'
    Json_save_PATH = '../data/NICE1/NICE1/coco_int/test/annotations/NICE_test.json' #json输出路径
    iris_Mask_PATH = '../data/NICE1/NICE1/coco_int/test/iris_edge_mask/'    #mask文件路径
    pupil_Mask_PATH = '../data/NICE1/NICE1/coco_int/test/pupil_edge_mask/'


    def get_contour(img):
        """获取连通域

        :param img: 输入图片
        :return: 最大连通域
        """
        # 灰度化, 二值化, 连通域分析
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

        img_contour, contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #findContours采到的轮廓点之间相邻的举例不大于一个像素

        return img_gray, contours[0]

    #取得目标轮廓像素点并采样
    def get_sampled_point_for_instancePoly(img_Path,Num_sample):
        # 1.导入图片
        img_gray = cv2.imread(img_Path)
        # 2.获取连通域
        img_gray, contour = get_contour(img_gray)
        # # 3.轮廓外边缘点
        # mask_out_edge = np.zeros(img_gray.shape, np.float32)
        # cv2.drawContours(mask_out_edge, [contour], 0, 255, 2)
        # pixel_point1 = cv2.findNonZero(mask_out_edge)
        # # 4.轮廓内部  (这里内部轮廓没处理，snake可以从外部收缩也可以从内部收缩，所以这个其实是可以用的，暂且留着)
        # mask_inside = np.zeros(img_gray.shape, np.float32)
        # cv2.drawContours(mask_inside, [contour], 0, 255, -1)
        # pixel_point2 = cv2.findNonZero(mask_inside)
        # # 5.根据期待得到的采样点个数采样
        # Num_pixel_point1 = pixel_point1.shape[0]
        # Num_pixel_point2 = pixel_point2.shape[0]
        # sample_pixel_point1 = np.array((Num_sample,1,2))
        # Sample_pixel_point2 = np.empty_like(sample_pixel_point1)
        # print("number of points:{}, \t {}".format(Num_pixel_point1,Num_pixel_point2))
        # if Num_sample > Num_pixel_point1:
        #     print("# num out of point1 length, return arange:", end=" ")
        # else:
        #     output = np.array([], dtype=pixel_point1.dtype)
        #     seg = len(pixel_point1) / Num_sample
        #     for n in range(Num_sample):
        #         if int(seg * (n + 1)) >= len(pixel_point1):
        #             output = np.append(output, pixel_point1[-1])
        #         else:
        #             output = np.append(output, pixel_point1[int(seg * n)])
        #         # print("output:{}".format(output))
        tmpoutput = []
        output=[]
        for tmpcontour in contour:
            tmpoutput.append(tmpcontour[0])
        for tmp in tmpoutput:
            output.append(tmp[0])
            output.append(tmp[1])
        return np.array(output)



    #读入bbox的信息并制作成字典，图片的长宽等信息不读入，不写入json
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
                if ext in dir[-len(ext):]:   #我们的拓展名是几个字符就改成几，如png 就是3；JPEG就是4
                    Filelist.append(dir)

        elif os.path.isdir(dir):
            for s in os.listdir(dir):
                newDir = os.path.join(dir, s)
                getFileList(newDir, Filelist, ext)

        return Filelist

    def as_dict(inisection):
        '''将ini的数据转换成字典格式'''

        d = dict(inisection)
        for k in d:
            d[k] = dict(d[k])
        return d




    # org_img_folder = os.path.join(Pic_PATH,'coco_int/train/image/')    #这里修改train和test
    # org_iris_ini_folder = os.path.join(Param_bbox_PATH,'train/iris_edge_bbox/')
    # org_pupil_ini_folder = os.path.join(Param_bbox_PATH,'train/pupil_edge_bbox/')
    # org_img_folder = os.path.join(Pic_PATH,'coco_int/val/image/')
    # org_iris_ini_folder = os.path.join(Param_bbox_PATH,'val/iris_edge_bbox/')
    # org_pupil_ini_folder = os.path.join(Param_bbox_PATH,'val/pupil_edge_bbox/')
    org_img_folder = os.path.join(Pic_PATH,'coco_int/test/image/')
    org_iris_ini_folder = os.path.join(Param_bbox_PATH,'test/iris_edge_bbox/')
    org_pupil_ini_folder = os.path.join(Param_bbox_PATH,'test/pupil_edge_bbox/')


    # 检索文件
    print("当前图片路径为：{}".format(org_img_folder))
    imglist = getFileList(org_img_folder, [], 'JPEG')
    iris_inilist = getFileList(org_iris_ini_folder, [], 'ini')
    pupil_inilist = getFileList(org_pupil_ini_folder, [], 'ini')
    print('本次执行检索到 ' + str(len(imglist)) + ' 个原始图片\n')
    print('本次执行检索到 ' + str(len(iris_inilist)) + ' 个iris ini文件\n')
    print('本次执行检索到 ' + str(len(pupil_inilist)) + ' 个pupil ini文件\n')
    round_num = 0
    image_id = 0
    # 主字典
    cat_info1 = {
                "id": 1, "name": "iris",
                "supercategory": "iris"

            }
    cat_info2 = {
        "id": 2, "name": "pupil",
        "supercategory": "pupil"

    }
    ret = {'images': [], 'annotations': [], "categories": []}
    for imgpath in imglist:
        round_num+=1
        print("完成第{}/{}次".format(round_num, len(imglist)))
        ininame = os.path.splitext(os.path.basename(imgpath))[0]
        iris_mask_file = os.path.join(iris_Mask_PATH,ininame+".png")
        pupil_mask_file = os.path.join(pupil_Mask_PATH,ininame+".png")
        img_file = os.path.join(org_img_folder, ininame+'.JPEG')
        #  实例化configParser对象
        config = configparser.ConfigParser()
        # -read读取ini文件
        if not os.path.isfile(org_iris_ini_folder+ininame+'.ini'):
            print(org_iris_ini_folder+ininame+'.ini')
            print("{}.ini 文件不存在".format(ininame))
        elif not os.path.isfile(org_pupil_ini_folder+ininame+'.ini'):
            print(org_pupil_ini_folder + ininame + '.ini')
            print("{}.ini 文件不存在".format(ininame))
        else:
            #读取外边界
            config.read(org_iris_ini_folder + ininame + '.ini')
            print(config._sections)
            iris_centerX = config.get('bbox', 'x')
            iris_centerY = config.get('bbox', 'y')
            iris_width = config.get('bbox', 'width')
            iris_height = config.get('bbox', 'height')
            iris_centerX = int(iris_centerX)
            iris_centerY = int(iris_centerY)
            iris_width = int(iris_width)
            iris_height = int(iris_height)
            config1 = as_dict(config)  #生成字典
            #读取内边界
            config.read(org_pupil_ini_folder + ininame + '.ini')
            print(config._sections)
            pupil_centerX = config.get('bbox', 'x')
            pupil_centerY = config.get('bbox', 'y')
            pupil_width = config.get('bbox', 'width')
            pupil_height = config.get('bbox', 'height')
            pupil_centerX = int(pupil_centerX)
            pupil_centerY = int(pupil_centerY)
            pupil_width = int(pupil_width)
            pupil_height = int(pupil_height)
            config2 = as_dict(config)  #生成字典

            #开始生成json格式文件
            # pascal_class_name = ["iris"]
            #
            # cat_ids = {cat: i + 1 for i, cat in enumerate(pascal_class_name)}
            #
            #
            # for i, cat in enumerate(pascal_class_name):
            #     cat_info.append({'name': cat, 'id': i + 1})

            # cat_info = {
            #     "id": round_num, "name": "iris",
            #     "supercategory": "iris"
            #
            # }



            bbox_iris = np.array([iris_centerX, iris_centerY, iris_width, iris_height], np.float)
            bbox_iris = bbox_iris.astype(np.int).tolist()
            bbox_pupil = np.array([pupil_centerX, pupil_centerY, pupil_width, pupil_height], np.float)
            bbox_pupil = bbox_pupil.astype(np.int).tolist()
            #得到设定采样点数目的采样点变量
            instance_poly_iris = get_sampled_point_for_instancePoly(iris_mask_file,128)
            instance_poly_pupil = get_sampled_point_for_instancePoly(pupil_mask_file,128)
            #得到图片的长宽尺寸
            img = cv2.imread(img_file)
            img_height = img.shape[1]
            img_width = img.shape[0]
            # print("instance poly shape:{}, \n data:{}".format(instance_poly.shape,instance_poly))
            img = {'file_name': str(ininame) + ".JPEG", 'id': round_num,
                   'height': img_height, 'width': img_width

                   }
            ann1 = {'area': iris_width*iris_height, 'iscrowd': 0, 'image_id':
                   round_num, 'bbox': bbox_iris,
                   'category_id': 1, 'id': int("00"+str(round_num)+"1"), 'ignore': 0,#'id': int(len(ret['annotations']) + 1)之前这么写的我不打算这么写了
                   'segmentation': [instance_poly_iris.tolist()]

            }
            ann2 = {'area': pupil_width*pupil_height, 'iscrowd': 0, 'image_id':
                   round_num, 'bbox': bbox_pupil,
                   'category_id': 2, 'id': int("00"+str(round_num)+"2"), 'ignore': 0,#'id': int(len(ret['annotations']) + 1)之前这么写的我不打算这么写了
                   'segmentation': [instance_poly_pupil.tolist()]

            }

            ret['images'].append(img)
            ret['annotations'].append(ann1)
            ret['annotations'].append(ann2)

            # ret['categories'].append(cat_info)
    # for i in range(1,100):
    #     cat_info = {
    #         "id": round_num+i, "name": "iris",
    #         "supercategory": "iris"
    #
    #     }
    #     ret['categories'].append(cat_info)
    ret['categories'].append(cat_info1)
    ret['categories'].append(cat_info2)
    json_fp = open(Json_save_PATH, 'w')
    json_str = json.dumps(ret)
    json_fp.write(json_str)
    json_fp.close()
