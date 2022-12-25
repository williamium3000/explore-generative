import numpy as np
import logging
import os
import torch
import os.path as osp
import torch
import mmcv
logs = set()
def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k  / batch_size)
        return res


def save_model(model, save_path, optimizer=None, scheduler=None, epoch=None):
    mmcv.mkdir_or_exist(osp.dirname(save_path))
    model = model.module if hasattr(model, 'module') else model
    to_save = {"model":model.state_dict()}
    if optimizer:
        to_save["optimizer"] = optimizer.state_dict()
    if scheduler:
        to_save["scheduler"] = scheduler.state_dict()
    if epoch:
        to_save["epoch"] = epoch
    torch.save(to_save, save_path)

def load_model(path, model, optimizer=None, scheduler=None):
    to_load = torch.load(path, map_location="cpu")
    if model:
        model.load_state_dict(to_load["model"])
    if optimizer:
        optimizer.load_state_dict(to_load["optimizer"])
    if scheduler:
        scheduler.load_state_dict(to_load["scheduler"])
    epoch = to_load.get("epoch", 0)
    return model, optimizer, scheduler, epoch