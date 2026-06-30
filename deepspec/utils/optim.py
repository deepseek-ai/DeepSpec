import torch
from torch.optim.lr_scheduler import CosineAnnealingLR as _CosineAnnealingLR
from torch.optim.lr_scheduler import LRScheduler as _LRScheduler


class TwoStageScheduler(_LRScheduler):
    def __init__(self, optimizer, after_scheduler: _LRScheduler, last_epoch=-1):
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch)

    def state_dict(self):
        state_dict = {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }
        if isinstance(state_dict["after_scheduler"], _LRScheduler):
            state_dict["after_scheduler_type"] = type(
                state_dict["after_scheduler"]
            ).__name__
            state_dict["after_scheduler_dict"] = state_dict[
                "after_scheduler"
            ].state_dict()
            del state_dict["after_scheduler"]
        else:
            raise NotImplementedError()
        return state_dict

    def load_state_dict(self, state_dict):
        self.after_scheduler.load_state_dict(state_dict["after_scheduler_dict"])
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if key not in ("after_scheduler_type", "after_scheduler_dict")
        }
        super().load_state_dict(state_dict)


class WarmupScheduler(TwoStageScheduler):
    def __init__(self, optimizer, warmup_epochs, after_scheduler, last_epoch=-1):
        self.warmup_epochs = int(warmup_epochs)
        super().__init__(optimizer, after_scheduler, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epochs:
            if not self.finished:
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished = True
                return self.base_lrs
            return self.after_scheduler.get_lr()

        return [(self.last_epoch + 1) / self.warmup_epochs * lr for lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished:
            if epoch is None:
                self.after_scheduler.step(None)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                self.after_scheduler.step(epoch - self.warmup_epochs)
                self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super().step(epoch)


class CosineAnnealingWarmupLR(WarmupScheduler):
    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        base_scheduler = _CosineAnnealingLR(
            optimizer,
            total_steps - warmup_steps,
            eta_min=eta_min,
            last_epoch=last_epoch,
        )
        super().__init__(optimizer, warmup_steps, base_scheduler, last_epoch=last_epoch)


class BF16Optimizer:
    # Adapted from SpecForge/specforge/optimizer.py:BF16Optimizer.
    def __init__(
        self,
        model,
        lr,
        total_steps,
        warmup_ratio,
        weight_decay=0.0,
    ):
        self.model = model
        self.model_params = [p for p in model.parameters() if p.requires_grad]
        self.fp32_params = [
            p.detach().clone().to(torch.float32) for p in self.model_params
        ]
        for param in self.fp32_params:
            param.requires_grad = True
        self.optimizer = torch.optim.AdamW(
            self.fp32_params, lr=lr, weight_decay=weight_decay
        )
        self.scheduler = CosineAnnealingWarmupLR(
            self.optimizer,
            total_steps=total_steps,
            warmup_steps=int(warmup_ratio * total_steps),
        )

    def step(self):
        with torch.no_grad():
            for model_param, master_param in zip(self.model_params, self.fp32_params):
                master_param.grad = (
                    model_param.grad.detach().to(torch.float32)
                    if model_param.grad is not None
                    else None
                )
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        with torch.no_grad():
            for model_param, master_param in zip(self.model_params, self.fp32_params):
                model_param.data.copy_(master_param.data.to(model_param.dtype))
                model_param.grad = None

    def state_dict(self):
        return {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "fp32_params": [param.detach().cpu() for param in self.fp32_params],
        }

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        fp32_params = state_dict["fp32_params"]
        for dst, src in zip(self.fp32_params, fp32_params):
            dst.data.copy_(src.to(dst.device))
        with torch.no_grad():
            for model_param, master_param in zip(self.model_params, self.fp32_params):
                model_param.data.copy_(master_param.data.to(model_param.dtype))

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]
