
from lib.utils import img_utils, data_utils
from lib.utils.snake import snake_config
import matplotlib.pyplot as plt
import numpy as np
import torch
from itertools import cycle
import os
import random

mean = snake_config.mean
std = snake_config.std


class Visualizer:
    def visualize_ex(self, output, batch, img_name):
        inp = img_utils.bgr_to_rgb(img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0))
        ex = output['py']
        ex = ex[-1] if isinstance(ex, list) else ex
        ex = ex.detach().cpu().numpy() * snake_config.down_ratio

        fig, ax = plt.subplots(1, figsize=(20, 10))
        fig.tight_layout()
        ax.axis('off')
        ax.imshow(inp)

        colors = np.array([
            [31, 119, 180],
            [255, 127, 14],
            [46, 160, 44],
            [214, 40, 39],
            [148, 103, 189],
            [140, 86, 75],
            [227, 119, 194],
            [126, 126, 126],
            [188, 189, 32],
            [26, 190, 207]
        ]) / 255.
        np.random.shuffle(colors)
        colors = cycle(colors)
        for i in range(len(ex)):
            color = next(colors).tolist()
            poly = ex[i]
            poly = np.append(poly, [poly[0]], axis=0)
            ax.plot(poly[:, 0], poly[:, 1], color=color)
        plt.savefig("/home/tianhao.lu/code/Deep_snake/snake/Result/Demo/eximg.jpg")
        plt.show()

    def visualize_training_box(self, output, batch, img_name):
        inp = img_utils.bgr_to_rgb(img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0))
        box = output['detection'][:, :4].detach().cpu().numpy() * snake_config.down_ratio
        ex = output['py']
        ex = ex[-1] if isinstance(ex, list) else ex
        ex = ex.detach().cpu().numpy() * snake_config.down_ratio

        #这里是查看显示结果的相关参数
        tmp_file = open('/home/tianhao.lu/code/Deep_snake/snake/Result/Contour/result.log', 'w', encoding='utf8')
        tmp_file.writelines("visualize training box -> inp：" + str(type(inp)) + "\n")
        tmp_file.writelines("visualize training box -> box：" + str(len(box)) + "\n")
        tmp_file.writelines("visualize training box -> ex:" + str(ex) + "\n")
        tmp_file.writelines("visualize training box -> detection:" + str(output['detection']) + "\n")
        # for tmp_data in train_loader:
        #     tmp_file.writelines("one train_loader data type:" + str(type(tmp_data)) + "\n")
        #     for key in tmp_data:
        #         tmp_file.writelines("one train_loader data key:" + str(key) + "\n")
        #         tmp_file.writelines("one train_loader data len:" + str(len(tmp_data[key])) + "\n")
        #     # tmp_file.writelines("one train_loader data:" + str(tmp_data) + "\n")
        #     break
        tmp_file.writelines(str("*************************************************************** \n"))
        tmp_file.close()
        fig, ax = plt.subplots(1, figsize=(20, 10))
        fig.tight_layout()
        ax.axis('off')
        ax.imshow(inp)

        colors = np.array([
            [31, 119, 180],
            [255, 127, 14],
            [46, 160, 44],
            [214, 40, 39],
            [148, 103, 189],
            [140, 86, 75],
            [227, 119, 194],
            [126, 126, 126],
            [188, 189, 32],
            [26, 190, 207]
        ]) / 255.
        np.random.shuffle(colors)
        colors = cycle(colors)
        for i in range(len(ex)):
            color = next(colors).tolist()
            poly = ex[i]
            poly = np.append(poly, [poly[0]], axis=0)
            ax.plot(poly[:, 0], poly[:, 1], color=color, linewidth=5)

            x_min, y_min, x_max, y_max = box[i]
            ax.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min], color='w', linewidth=0.5)

        # filename = random.randint(0,100000)  #这个是名称取十万以内的随机数
        filename = img_name
        # fig = plt.gcf()
        img_width = 400
        img_height = 300
        fig.set_size_inches(img_width / 96.0, img_height / 96.0)  # 输出width*height像素
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)  # 输出图像#边框设置
        plt.margins(0, 0)
        plt.savefig("/home/tianhao.lu/code/Deep_snake/snake/Result/coco_test_result/%s.png"%filename,dpi=96.0,pad_inches=0.0)
        # plt.savefig("/home/tianhao.lu/code/Deep_snake/snake/Result/Demo/trainboximg.jpg")
        # plt.show()

    def visualize(self, output, batch, img_name):
        self.visualize_ex(output, batch, img_name)
        self.visualize_training_box(output, batch, img_name)

