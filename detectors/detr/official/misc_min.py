"""Minimal stand-ins for detr/util/misc.py (single process, fixed-size batches without padding masks)."""
from typing import Optional

import torch
from torch import Tensor


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors, self.mask = tensors, mask

    def decompose(self):
        return self.tensors, self.mask


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k (detr/util/misc.py)"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_world_size():
    return 1


def is_dist_avail_and_initialized():
    return False


def nested_tensor_from_tensor_list(tensor_list):
    raise NotImplementedError("padding masks are not used in this framework (fixed 608 x 608 input)")


def interpolate(*args, **kwargs):
    return torch.nn.functional.interpolate(*args, **kwargs)


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def is_main_process():
    return True


def get_rank():
    return 0


def clean_state_dict(state_dict):
    return state_dict
