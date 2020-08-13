from .trainer import Trainer
import imp
import os


def _wrapper_factory(cfg, network):
    module = '.'.join(['lib.train.trainers', cfg.task])
    path = os.path.join('lib\train\trainers', cfg.task+'.py')
    network_wrapper = imp.load_source(module, path).NetworkWrapper(network)    #名字叫网络包装，感觉实际就是计算了些损失函数，然后放在了loss和output里面
    return network_wrapper


def make_trainer(cfg, network):
    network = _wrapper_factory(cfg, network)
    return Trainer(network)
