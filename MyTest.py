from lib.config import cfg, args
import numpy as np
import os
import argparse
import sys
sys.path.append("")
from lib.config.yacs import CfgNode as CN
from pycocotools.coco import COCO

#初始化cfg的参数
cfg = CN()

# model
cfg.model = 'hello'
cfg.model_dir = 'data/model'

# network
cfg.network = 'dla_34'

# network heads
cfg.heads = CN()

# task
cfg.task = ''

# gpus
cfg.gpus = [1]

# if load the pretrained network
cfg.resume = True


# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
cfg.train = CN()

cfg.train.dataset = 'CocoTrain'
cfg.train.epoch = 140
cfg.train.num_workers = 8

# use adam as default
cfg.train.optim = 'adam'
cfg.train.lr = 1e-4
cfg.train.weight_decay = 5e-4

cfg.train.warmup = False
cfg.train.scheduler = ''
cfg.train.milestones = [80, 120, 200, 240]
cfg.train.gamma = 0.5

cfg.train.batch_size = 4

# test
cfg.test = CN()
cfg.test.dataset = 'CocoVal'
cfg.test.batch_size = 1
cfg.test.epoch = -1

# recorder
cfg.record_dir = 'data/record'

# result
cfg.result_dir = 'data/result'

# evaluation
cfg.skip_eval = False

cfg.save_ep = 5
cfg.eval_ep = 5

cfg.use_gt_det = False

# -----------------------------------------------------------------------------
# snake
# -----------------------------------------------------------------------------
cfg.ct_score = 0.3
cfg.demo_path = ''

def parse_cfg(cfg, args):
    if len(cfg.task) == 0:
        raise ValueError('task must be specified')

    # assign the gpus
    # os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(gpu) for gpu in cfg.gpus])
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    CUDA_VISIBLE_DEVICES = 1

    cfg.det_dir = os.path.join(cfg.model_dir, cfg.task, args.det)

    # assign the network head conv
    cfg.head_conv = 64 if 'res' in cfg.network else 256

    cfg.model_dir = os.path.join(cfg.model_dir, cfg.task, cfg.model)
    cfg.record_dir = os.path.join(cfg.record_dir, cfg.task, cfg.model)
    cfg.result_dir = os.path.join(cfg.result_dir, cfg.task, cfg.model)


def make_cfg(args):
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    parse_cfg(cfg, args)
    return cfg


def run_dataset():
    from lib.datasets import make_data_loader
    import tqdm

    cfg.train.num_workers = 0
    data_loader = make_data_loader(cfg, is_train=False)
    for batch in tqdm.tqdm(data_loader):
        pass


def run_network():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    import time

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    total_time = 0
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            network(batch['inp'])
            torch.cuda.synchronize()
            total_time += time.time() - start
    print(total_time / len(data_loader))


def run_evaluate():
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    import tqdm
    import torch
    from lib.networks import make_network
    from lib.utils.net_utils import load_network

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    for batch in tqdm.tqdm(data_loader):
        inp = batch['inp'].cuda()
        with torch.no_grad():
            output = network(inp)
        evaluator.evaluate(output, batch)
    evaluator.summarize()


def run_visualize():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    from lib.visualizers import make_visualizer

    network = make_network(cfg).cuda()
    print("model dir:{}".format(cfg.model_dir))
    load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    visualizer = make_visualizer(cfg)
    ann_file = 'data/NICE1/NICE1/coco_int/test/annotations/NICE_test.json'  # 这里是Test的json，因为这里使用的就是test的集合
    coco = COCO(ann_file)
    for batch in tqdm.tqdm(data_loader):  #tqdm是个进度条
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = network(batch['inp'], batch)
        # print("batch  ->{}".format(batch)) #batch是对的，里面还有img_id等相关信息
        # print("batch['meta']  ->{}".format(batch['meta'])) #batch['meta']是对的
        img_info = batch['meta']
        img_id = img_info['img_id']  #这个img_id用来根据id读取img名称从而确定输出图片名称的
        img_scale = img_info['scale']  # 这个img_scale是用来读取尺寸从而改变图像尺寸的
        # print("img id ->{}".format(img_id))
        # print("img scale ->{}".format(img_scale))
        img_name=coco.loadImgs(int(img_id))[0]['file_name']
        img_name,_ = os.path.splitext(img_name)
        # ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=0)  #两种方法，一种获得anno，一种直接loadimg获得filename
        # anno = coco.loadAnns(ann_ids)
        visualizer.visualize(output, batch, img_name)
        tmp_file = open('/home/tianhao.lu/code/Deep_snake/snake/Result/Contour/Output_check.log', 'w', encoding='utf8')
        tmp_file.writelines("Output -> ：" + str(output) + "\n")
        tmp_file.writelines("batch -> ：" + str(batch) + "\n")
        # for tmp_data in train_loader:
        #     tmp_file.writelines("one train_loader data type:" + str(type(tmp_data)) + "\n")
        #     for key in tmp_data:
        #         tmp_file.writelines("one train_loader data key:" + str(key) + "\n")
        #         tmp_file.writelines("one train_loader data len:" + str(len(tmp_data[key])) + "\n")
        #     # tmp_file.writelines("one train_loader data:" + str(tmp_data) + "\n")
        #     break
        tmp_file.writelines(str("*************************************************************** \n"))
        tmp_file.close()
        # print("output write finish one time!")


def run_sbd():
    from tools import convert_sbd
    convert_sbd.convert_sbd()


def run_demo():
    from tools import demo
    demo.demo()


if __name__ == '__main__':
    # 定义一下arg
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", default="./configs/Mycoco_test.yaml", type=str)
    parser.add_argument('--test', action='store_true', dest='test', default=False)
    parser.add_argument("--type", type=str, default="")
    parser.add_argument('--det', type=str, default='')
    parser.add_argument('-f', type=str, default='')
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    print(args)
    cfg = make_cfg(args)

    args.type = 'visualize'
    globals()['run_'+args.type]()
