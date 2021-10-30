# This file is modified from https://github.com/traveller59/second.pytorch

import math
from functools import partial

import numpy as np
import torch.optim.lr_scheduler as lr_sched

from .fastai_optim import OptimWrapper


class LRSchedulerStep(object):
    def __init__(self, fai_optimizer: OptimWrapper, total_step, lr_phases,
                 mom_phases):
        # if not isinstance(fai_optimizer, OptimWrapper):
        #     raise TypeError('{} is not a fastai OptimWrapper'.format(
        #         type(fai_optimizer).__name__))
        self.optimizer = fai_optimizer # ptimWrapper over Adam
        self.total_step = total_step # 74240
        self.lr_phases = []

        for i, (start, lambda_func) in enumerate(lr_phases):
            if len(self.lr_phases) != 0: # self.lr_phases列表在起始时为0，即空列表
                assert self.lr_phases[-1][0] < start # 0 < 0.4
            if isinstance(lambda_func, str):
                lambda_func = eval(lambda_func)
            if i < len(lr_phases) - 1: # lr_phases的长度为2
                self.lr_phases.append((int(start * total_step), int(lr_phases[i + 1][0] * total_step), lambda_func))
            else:
                self.lr_phases.append((int(start * total_step), total_step, lambda_func))
        assert self.lr_phases[0][0] == 0
        """
        self.lr_phases =
        [(0,
          29696,
          functools.partial(<function annealing_cos at 0x7fe892744f80>, 0.00030000000000000003, 0.003)),
          (29696,
          74240,
          functools.partial(<function annealing_cos at 0x7fe892744f80>, 0.003, 3.0000000000000004e-08))]
        """
        
        self.mom_phases = []
        for i, (start, lambda_func) in enumerate(mom_phases):
            if len(self.mom_phases) != 0:
                assert self.mom_phases[-1][0] < start
            if isinstance(lambda_func, str):
                lambda_func = eval(lambda_func)
            if i < len(mom_phases) - 1:
                self.mom_phases.append((int(start * total_step), int(mom_phases[i + 1][0] * total_step), lambda_func))
            else:
                self.mom_phases.append((int(start * total_step), total_step, lambda_func))
        assert self.mom_phases[0][0] == 0
        """
        mom_phases = 
        [(0,
          29696,
          functools.partial(<function annealing_cos at 0x7fe892744f80>, 0.95, 0.85)),
          (29696,
          74240,
          functools.partial(<function annealing_cos at 0x7fe892744f80>, 0.85, 0.95))]
        """

    def step(self, step):
        # 根据start判断起点分段更新lr和momentum
        for start, end, func in self.lr_phases:
            if step >= start:
                self.optimizer.lr = func((step - start) / (end - start))
        for start, end, func in self.mom_phases:
            if step >= start:
                self.optimizer.mom = func((step - start) / (end - start))


def annealing_cos(start, end, pct):
    # print(pct, start, end)
    "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start - end) / 2 * cos_out


class OneCycle(LRSchedulerStep):
    def __init__(self, fai_optimizer, total_step, lr_max, moms, div_factor,
                 pct_start):
        self.lr_max = lr_max # 0.003
        self.moms = moms # [0.95, 0.85]
        self.div_factor = div_factor # 10 初始学习率= max_lr / div_factor
        self.pct_start = pct_start # 0.4 学习率上升部分
        a1 = int(total_step * self.pct_start) # 74240 * 0.4 = 29696
        a2 = total_step - a1 # 44544
        low_lr = self.lr_max / self.div_factor # 0.003
        # 两个部分的start和end相反,代用annealing_cos生成趋势上升和下降的曲线，且限制在start和end之间
        lr_phases = ((0, partial(annealing_cos, low_lr, self.lr_max)), # （0.0003，0.003）
                     (self.pct_start,
                      partial(annealing_cos, self.lr_max, low_lr / 1e4))) # (0.003, 3e-8)
        mom_phases = ((0, partial(annealing_cos, *self.moms)),
                      (self.pct_start, partial(annealing_cos,
                                               *self.moms[::-1])))
        fai_optimizer.lr, fai_optimizer.mom = low_lr, self.moms[0] # 0.003，0.95 
        super().__init__(fai_optimizer, total_step, lr_phases, mom_phases) # adam, 74240, lr_phases， mom_phases


class CosineWarmupLR(lr_sched._LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max # lr的变化是周期性的，T_max是周期的1/2 
        self.eta_min = eta_min # lr的最小值，默认为0
        # 最后一个epoch的index，如果是训练了很多个epoch后中断了，继续训练，这个值就等于加载的模型的epoch
        # 默认为-1表示从头开始训练，即从epoch=1开始
        super(CosineWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # 余弦退火公式
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]


class FakeOptim:
    def __init__(self):
        self.lr = 0
        self.mom = 0


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    opt = FakeOptim()  # 3e-3, wd=0.4, div_factor=10
    schd = OneCycle(opt, 100, 3e-3, (0.95, 0.85), 10.0, 0.1)

    lrs = []
    moms = []
    for i in range(100):
        schd.step(i)
        lrs.append(opt.lr)
        moms.append(opt.mom)
    plt.plot(lrs)
    # plt.plot(moms)
    plt.show()
    plt.plot(moms)
    plt.show()
