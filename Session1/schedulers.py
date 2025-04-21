import math
from torch.optim import Optimizer

def cosine_decay(epoch: int, epochs: int):
    return 0.5 * (1 + math.cos(math.pi * epoch / epochs))

def linear_decay(epoch: int, epochs: int) -> float:
    return 1 - epoch / epochs

class LearningRateScheduler:
    def __init__(self, lr_base, optimizer: Optimizer, get_decay_rate, start_epoch = 0):
        self.lr_base = lr_base
        self.optimizer = optimizer
        self.start_epoch = start_epoch
        self.get_decay_rate = get_decay_rate
        self.current_epoch = 0
        self.current_lr = 0
        
    def step(self):
        # gradually increase LR up to target when warmup epochs are given
        if self.current_epoch < self.start_epoch:
            self.current_lr = self.lr_base * (self.current_epoch + 1) / self.start_epoch
        # o/w compute decaying rate
        else:
            # no decay rate func given => use constant LR
            decay_rate = self.get_decay_rate(self.current_epoch - self.start_epoch) if self.get_decay_rate else 1.0
            self.current_lr = self.lr_base * decay_rate
        
        # source: https://discuss.pytorch.org/t/how-could-i-design-my-own-optimizer-scheduler/26810
        # ? update all parameter groups of the optimizer for all layers ?
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr
        
        self.current_epoch += 1

    def get(self):
        return self.current_lr
