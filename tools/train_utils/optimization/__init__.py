from functools import partial

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched

from .fastai_optim import OptimWrapper
from .learning_schedules_fastai import CosineWarmupLR, OneCycle


def build_optimizer(model, optim_cfg):
    # 如果采用adam优化器，则调用optim的Adam初始化
    if optim_cfg.OPTIMIZER == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY)
    # 如果采用sdg优化器，则调用optim的sgd初始化
    elif optim_cfg.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY,
            momentum=optim_cfg.MOMENTUM
        )
    # 如果采用adam_onecycle优化器
    elif optim_cfg.OPTIMIZER == 'adam_onecycle':
        
        def children(m: nn.Module):
            # 取出该模块中的各个子模块，组成list
            return list(m.children())

        def num_children(m: nn.Module) -> int:
            # 返回该模块子模块的长度
            return len(children(m))

        # 递归调用将基础层（Linear，Conv2d，BatchNorm2d，ReLU，ConvTranspose2d等）堆叠成list
        # a = [[1], [2], [3], [4], [5]] 
        # sum(a, []) # [1, 2, 3, 4, 5]
        flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
        # 将所有基础层堆叠的list连接成序列
        get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]

        optimizer_func = partial(optim.Adam, betas=(0.9, 0.99)) # 初始化adam优化器
        optimizer = OptimWrapper.create(
            optimizer_func, 3e-3, get_layer_groups(model), wd=optim_cfg.WEIGHT_DECAY, true_wd=True, bn_wd=True
        ) # wd:weight decay
    else:
        raise NotImplementedError

    return optimizer


def build_scheduler(optimizer, total_iters_each_epoch, total_epochs, last_epoch, optim_cfg):
    """
    构建学习率调度器：三种方式adam_onecycle、LambdaLR、CosineWarmupLR
    Args:
        optimizer:优化器
        total_iters_each_epoch：一个epoch的迭代次数:982
        total_epoch:总共的epoch数:80
        last_epoch:上一次的epoch_id
        optim_cfg:优化配置
    """
    decay_steps = [x * total_iters_each_epoch for x in optim_cfg.DECAY_STEP_LIST] # [35, 45] --> [32480, 41760]
    # 自定义学习率调度函数
    def lr_lbmd(cur_epoch):
        # 当前学习率设置为1
        cur_decay = 1
        for decay_step in decay_steps:
            # 如果当前的epoch数>=节点值
            if cur_epoch >= decay_step:
                # 更新学习率
                cur_decay = cur_decay * optim_cfg.LR_DECAY # LR_DECAY: 0.1
        # 防止学习率过小
        return max(cur_decay, optim_cfg.LR_CLIP / optim_cfg.LR) # LR_CLIP: 0.0000001

    lr_warmup_scheduler = None
    total_steps = total_iters_each_epoch * total_epochs # 74240
    # 构建adam_onecycle学习率调度器
    if optim_cfg.OPTIMIZER == 'adam_onecycle':
        lr_scheduler = OneCycle(
            optimizer, total_steps, optim_cfg.LR, list(optim_cfg.MOMS), optim_cfg.DIV_FACTOR, optim_cfg.PCT_START
        ) # LR: 0.003, MOMS: [0.95, 0.85],DIV_FACTOR: 10, PCT_START: 0.4 
    else:
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)
        # 热身：在刚刚开始训练时以很小的学习率进行训练，使得网络熟悉数据，随着训练的进行学习率慢慢变大，
        # 到了一定程度，以设置的初始学习率进行训练，接着过了一些inter后，学习率再慢慢变小；学习率变化：上升——平稳——下降；
        if optim_cfg.LR_WARMUP:
            lr_warmup_scheduler = CosineWarmupLR(
                optimizer, T_max=optim_cfg.WARMUP_EPOCH * len(total_iters_each_epoch),
                eta_min=optim_cfg.LR / optim_cfg.DIV_FACTOR
            ) # WARMUP_EPOCH: 1

    return lr_scheduler, lr_warmup_scheduler
