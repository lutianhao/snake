import sys
import os
sys.path.append("")
import matplotlib.pyplot as plt
import imutils
import configparser
import numpy as np
import cv2
from lib.config import cfg, args
from lib.networks import make_network
from lib.train import make_trainer, make_optimizer, make_lr_scheduler, make_recorder, set_lr_scheduler
from lib.datasets import make_data_loader
from lib.utils.net_utils import load_model, save_model, load_network
from lib.evaluators import make_evaluator
from lib.config.yacs import CfgNode as CN
import argparse


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

cfg.save_ep = 100
cfg.eval_ep = 100

cfg.use_gt_det = False

# -----------------------------------------------------------------------------
# snake
# -----------------------------------------------------------------------------
cfg.ct_score = 0.05
cfg.demo_path = ''



def train(cfg, network):
    trainer = make_trainer(cfg, network)
    optimizer = make_optimizer(cfg, network)
    scheduler = make_lr_scheduler(cfg, optimizer)
    recorder = make_recorder(cfg)
    evaluator = make_evaluator(cfg)

    begin_epoch = load_model(network, optimizer, scheduler, recorder, cfg.model_dir, resume=cfg.resume)
    # begin_epoch = 0  #如果要继续训练那么请注释这一行
    # set_lr_scheduler(cfg, scheduler)
    # print("before train loader")
    train_loader = make_data_loader(cfg, is_train=True)  #到这里才读取的数据
    # print("under train loader")
    val_loader = make_data_loader(cfg, is_train=False)

    # #这里是查看train_loader的相关参数个结构
    # tmp_file = open('/home/tianhao.lu/code/Deep_snake/snake/Result/Contour/contour.log', 'w')
    # tmp_file.writelines("train_loader type:" + str(type(train_loader)) + "\n")
    # tmp_file.writelines("train_loader len:" + str(len(train_loader)) + "\n")
    # tmp_file.writelines("train_loader data:" + str(train_loader) + "\n")
    # for tmp_data in train_loader:
    #     tmp_file.writelines("one train_loader data type:" + str(type(tmp_data)) + "\n")
    #     for key in tmp_data:
    #         tmp_file.writelines("one train_loader data key:" + str(key) + "\n")
    #         tmp_file.writelines("one train_loader data len:" + str(len(tmp_data[key])) + "\n")
    #     # tmp_file.writelines("one train_loader data:" + str(tmp_data) + "\n")
    #     break
    # tmp_file.writelines(str("*************************************************************** \n"))
    # tmp_file.close()

    for epoch in range(begin_epoch, cfg.train.epoch):
        recorder.epoch = epoch
        trainer.train(epoch, train_loader, optimizer, recorder)
        scheduler.step()      #optimizer.step()模型才会更新，scheduler.step()是用来调整lr的

        if (epoch + 1) % cfg.save_ep == 0:
            save_model(network, optimizer, scheduler, recorder, epoch, cfg.model_dir)

        if (epoch + 1) % cfg.eval_ep == 0:
            trainer.val(epoch, val_loader, evaluator, recorder)

    return network

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


def main():
    #定义一下arg
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

    #开始创建网络和训练
    network = make_network(cfg)
    train(cfg, network)


if __name__ == "__main__":
    main()