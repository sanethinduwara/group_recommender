from torch.optim.lr_scheduler import _LRScheduler
import math


class CyclicLR(_LRScheduler):
    
    def __init__(self, optimizer, schedule, last_epoch=-1):
        assert callable(schedule)
        self.schedule = schedule
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.schedule(self.last_epoch, lr) for lr in self.base_lrs]

    @staticmethod
    def cosine(t_max, eta_min=0):
        def scheduler(epoch, base_lr):
            t = epoch % t_max
            return eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * t / t_max)) / 2

        return scheduler
