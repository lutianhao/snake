from .transforms import make_transforms
from . import samplers
from .dataset_catalog import DatasetCatalog
import torch
import torch.utils.data
import imp
import os
from .collate_batch import make_collator


torch.multiprocessing.set_sharing_strategy('file_system')


def _dataset_factory(data_source, task):
    module = '.'.join(['lib.datasets', data_source, task])
    path = os.path.join('lib/datasets', data_source, task+'.py')
    dataset = imp.load_source(module, path).Dataset
    return dataset


def make_dataset(cfg, dataset_name, transforms, is_train=True):
    args = DatasetCatalog.get(dataset_name)
    print("args is: {}".format(args))
    data_source = args['id']
    print("data source :{}".format(data_source))
    dataset = _dataset_factory(data_source, cfg.task)
    del args['id']
    # args['cfg'] = cfg
    # args['transforms'] = transforms
    # args['is_train'] = is_train
    dataset = dataset(**args)
    print("Dataset : {}".format(len(dataset)))
    return dataset


def make_data_sampler(dataset, shuffle):
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(cfg, sampler, batch_size, drop_last, max_iter):
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last)
    if max_iter != -1:
        batch_sampler = samplers.IterationBasedBatchSampler(batch_sampler, max_iter)
    return batch_sampler


def make_data_loader(cfg, is_train=True, is_distributed=False, max_iter=-1):
    if is_train:
        batch_size = cfg.train.batch_size
        shuffle = True
        drop_last = False
    else:
        batch_size = cfg.test.batch_size
        shuffle = True if is_distributed else False
        drop_last = False

    dataset_name = cfg.train.dataset if is_train else cfg.test.dataset
    print("dataset name:{}".format(dataset_name))
    transforms = make_transforms(cfg, is_train)    #这里是给数据集的均值和标准差
    dataset = make_dataset(cfg, dataset_name, transforms, is_train)
    #查看一下某一个样本
    # tmp_feature, tmp_label = dataset[3] #注意这是MNIST的，coco不能这么导入因为一条样本的信息不一样
    # print(tmp_feature.shape,tmp_label)
    print(len(dataset))
    #
    sampler = make_data_sampler(dataset, shuffle)      #这个是采样，乱序采样
    batch_sampler = make_batch_data_sampler(cfg, sampler, batch_size, drop_last, max_iter)
    num_workers = cfg.train.num_workers
    collator = make_collator(cfg)    #这个collator不是很明白，翻译过来是校对整理
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collator
    )

    return data_loader
