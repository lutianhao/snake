import json
import numpy as np
import os
import sys
import cv2
import configparser


if __name__ == '__main__':                            #设置相关路径
    Pic_PATH = '../data/NICE1/NICE1/' #图片路径，注意修改下面的train 或者test路径
    Param_bbox_PATH = '../data/NICE1/NICE1/coco/' #bbox路径
    Json_save_PATH = '../data/NICE1/NICE1/coco/train/annotations/' #json输出路径

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




    org_img_folder = os.path.join(Pic_PATH,'coco/train/image/')    #这里修改train和test
    org_ini_folder = os.path.join(Param_bbox_PATH,'train/iris_edge_bbox/')    #这里修改train和test
    # org_img_folder = os.path.join(Pic_PATH,'coco/test/iris_edge_bbox')
    # org_ini_folder = os.path.join(Param_bbox_PATH,'coco/test/iris_edge_bbox')

    # 检索文件
    print("当前图片路径为：{}".format(org_img_folder))
    imglist = getFileList(org_img_folder, [], 'JPEG')
    inilist = getFileList(org_ini_folder, [], 'ini')
    print('本次执行检索到 ' + str(len(imglist)) + ' 个原始图片\n')
    print('本次执行检索到 ' + str(len(inilist)) + ' 个ini文件\n')
    round_num = 0
    image_id = 0
    for imgpath in imglist:
        round_num+=1
        print("完成第{}/{}次".format(round_num, len(imglist)))
        ininame = os.path.splitext(os.path.basename(imgpath))[0]
        #  实例化configParser对象
        config = configparser.ConfigParser()
        # -read读取ini文件
        if not os.path.isfile(org_ini_folder+ininame+'.ini'):
            print(org_ini_folder+ininame+'.ini')
            print("{}.ini 文件不存在".format(ininame))
        else:
            config.read(org_ini_folder + ininame + '.ini')
            print(config._sections)
            iris_centerX = config.get('bbox', 'x')
            iris_centerY = config.get('bbox', 'y')
            iris_wdith = config.get('bbox', 'width')
            iris_height = config.get('bbox', 'height')
            x = int(iris_centerX)
            y = int(iris_centerY)
            w = int(iris_wdith)
            h = int(iris_height)
            config = as_dict(config)  #生成字典

            #开始生成json格式文件
            pascal_class_name = ["iris"]

            cat_ids = {cat: i + 1 for i, cat in enumerate(pascal_class_name)}

            cat_info = []
            for i, cat in enumerate(pascal_class_name):
                cat_info.append({'name': cat, 'id': i + 1})

            # 主字典
            ret = {'images': [], 'annotations': [], "categories": cat_info}

            bbox = np.array([x, y, w, h], np.float)
            bbox = bbox.astype(np.int).tolist()
            ann = {'image_id': image_id,
                   'id': int(len(ret['annotations']) + 1),
                   'bbox': bbox,
                    'ignore': 0
            }
            ret['annotations'].append(ann)
            image_id += 1
            #下面这段不知为什么输出的是重复的多次annotation
            # for key in config:
            #     lines = config[key]
            #     image_info = {'file_name': key,
            #                   'id': int(image_id)}
            #     # images data
            #     ret['images'].append(image_info)
            #
            #     # anno data
            #     for line in lines:
            #         bbox = np.array([x,y,w,h], np.float)
            #         bbox = bbox.astype(np.int).tolist()
            #         ann = {'image_id': image_id,
            #                'id': int(len(ret['annotations']) + 1),
            #                'bbox': bbox,
            #                 'ignore': 0
            #         }
            #         ret['annotations'].append(ann)
            #     image_id += 1

            out_path = Json_save_PATH+ ininame + '.json'
            json.dump(ret, open(out_path, 'w'))
