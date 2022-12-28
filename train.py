import argparse
import logging
import os
import pprint
import time
from tempfile import TemporaryDirectory

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import tqdm
from mmcv import Config
from torchvision.transforms import autoaugment, transforms
import torchvision
from torchvision.datasets import DatasetFolder
from torchvision.utils import save_image

from utils.utils import count_params, init_log, accuracy, FolderDataset
from utils.dist import setup_distributed
from utils.inception_score import inception_score
from pytorch_fid.fid_score import calculate_fid_given_paths
from models.builder import build_model
from datasets.builder import build_trainset
from eval import evaluate, sample

parser = argparse.ArgumentParser(description='train domain generalization (oracle)')
parser.add_argument('--cfg', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True, choices=["cifar10", "cifar100", "celeba", "imagenet", "mnist"])
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--eval_freq', default=1, type=int, help="do not eval if -1")

     
def main():
    args = parser.parse_args()
    cfg = Config.fromfile(args.cfg)
    
    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, word_size = setup_distributed(port=args.port)

    if rank == 0:
        logger.info('{}\n'.format(pprint.pformat(cfg)))

    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    # model = VAE(in_channels=3, latent_dim=512, hidden_dims=[32, 128, 512])
    model = build_model(cfg)
    
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    param_groups = model.parameters()
    opt_name = cfg["optimizer"].lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            param_groups,
            lr=cfg['lr'],
            momentum=cfg['momentum'],
            weight_decay=cfg['weight_decay'],
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            param_groups, lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'], eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(param_groups, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(param_groups, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    else:
        raise RuntimeError(f"Invalid optimizer {opt_name}. Only SGD, RMSprop and AdamW are supported.")
    
    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    model.cuda(local_rank)
    
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=False)

    trainset = build_trainset(args)
    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'],
                             pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler)

    if "scheduler" in cfg:
        if cfg["scheduler"]["name"] == "StepLR":
            scheduler = lr_scheduler.StepLR(optimizer, **cfg["scheduler"]["kwargs"])
        elif cfg["scheduler"]["name"] == "cosineannealinglr":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg['epochs'] if cfg["scheduler"]["by_epoch"] else cfg['epochs'] * len(trainloader), **cfg["scheduler"]["kwargs"]
            )
        elif cfg["scheduler"]["name"] == "exponentiallr":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **cfg["scheduler"]["kwargs"])

    for epoch in range(cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.6f}'.format(
                epoch, optimizer.param_groups[0]['lr']))

        model.train()
        total_loss = 0.0
        total_num = 0.0
        data_time = 0.0
        model_time = 0.0
        trainsampler.set_epoch(epoch)
        time0 = time.time()
        for i, (img, mask) in enumerate(trainloader):
            
            time1 = time.time()
            
            img, mask = img.cuda(), mask.cuda()
            loss = model(x=img)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(param_groups, 1.0)
            optimizer.step()
            
            time2 = time.time()
            
            bs = img.size(0)
            total_num += bs
            total_loss += loss * bs
            
            data_time += time1 - time0
            model_time += time2 - time1
            
            if (i % 50 == 0) and (rank == 0):
                logger.info('Iters: {} / {}, data time: {:.2f}, model time: {:.2f}, Total loss: {:.4f}'.format(
                    i, len(trainloader), 
                    data_time / (i + 1), model_time / (i + 1),
                    total_loss.item() / total_num))
            
            if "scheduler" in cfg and not cfg["scheduler"]["by_epoch"]:
                scheduler.step()
            
            time0 = time.time()

        if "scheduler" in cfg and cfg["scheduler"]["by_epoch"]:
            scheduler.step()
        
        if args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0:
            IS, IS_std, FID = evaluate(model, cfg, sample_num=1000, local_rank=rank, word_size=word_size)
            if rank == 0:
                logger.info('***** Evaluation ***** >>>> IS: {:.2f} IS std : {:.2f} FID : {:.2f}\n'.format(
                    IS, IS_std, FID))
            
        if rank == 0:
            torch.save(model.module.state_dict(),
                       os.path.join(args.save_path, 'latest.pth'))
            # visualization
            with torch.no_grad():
                sampled_imgs = model(num_samples=64)
                sampled_imgs = sampled_imgs * 0.5 + 0.5
                save_image(sampled_imgs, os.path.join(args.save_path, f"visualization.jpg"), nrow=8)


if __name__ == '__main__':
    main()
