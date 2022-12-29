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

    opt_name = cfg["optimizer"].lower()
    if opt_name.startswith("sgd"):
        optimizer_g = torch.optim.SGD(
            model.G.parameters(),
            lr=cfg['lr'],
            momentum=cfg['momentum'],
            weight_decay=cfg['weight_decay'],
            nesterov="nesterov" in opt_name,
        )
        optimizer_d = torch.optim.SGD(
            model.D.parameters(),
            lr=cfg['lr'],
            momentum=cfg['momentum'],
            weight_decay=cfg['weight_decay'],
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer_g = torch.optim.RMSprop(
            model.G.parameters(), lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'], eps=0.0316, alpha=0.9
        )
        optimizer_d = torch.optim.RMSprop(
            model.D.parameters(), lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'], eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer_g = torch.optim.AdamW(model.G.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        optimizer_d = torch.optim.AdamW(model.D.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    elif opt_name == "adam":
        optimizer_g = torch.optim.Adam(model.G.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'], betas=cfg['betas'])
        optimizer_d = torch.optim.Adam(model.D.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'], betas=cfg['betas'])
    else:
        raise RuntimeError(f"Invalid optimizer {opt_name}. Only SGD, RMSprop and AdamW are supported.")
    
    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    model.cuda(local_rank)
    
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=False)

    trainset = build_trainset(args, cfg)
    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'],
                             pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler)

    if "scheduler" in cfg:
        if cfg["scheduler"]["name"] == "StepLR":
            scheduler_g = lr_scheduler.StepLR(optimizer_g, **cfg["scheduler"]["kwargs"])
            scheduler_d = lr_scheduler.StepLR(optimizer_d, **cfg["scheduler"]["kwargs"])
        elif cfg["scheduler"]["name"] == "cosineannealinglr":
            scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_g, T_max=cfg['epochs'] if cfg["scheduler"]["by_epoch"] else cfg['epochs'] * len(trainloader), **cfg["scheduler"]["kwargs"]
            )
            scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_d, T_max=cfg['epochs'] if cfg["scheduler"]["by_epoch"] else cfg['epochs'] * len(trainloader), **cfg["scheduler"]["kwargs"]
            )
        elif cfg["scheduler"]["name"] == "exponentiallr":
            scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer_g, **cfg["scheduler"]["kwargs"])
            scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer_d, **cfg["scheduler"]["kwargs"])

    for epoch in range(cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.6f}'.format(
                epoch, optimizer_g.param_groups[0]['lr']))

        model.train()
        total_loss_g = 0.0
        total_loss_d = 0.0
        total_num = 0.0
        data_time = 0.0
        model_time = 0.0
        trainsampler.set_epoch(epoch)
        time0 = time.time()
        for i, (img, _) in enumerate(trainloader):
            
            time1 = time.time()
            
            img = img.cuda()
            loss_g, loss_d = model(x=img)

            optimizer_g.zero_grad()
            loss_g.backward()
            if "grad_clip" in cfg:
                torch.nn.utils.clip_grad_norm_(model.G.parameters(), cfg["grad_clip"])
            optimizer_g.step()
            
            optimizer_d.zero_grad()
            loss_d.backward()
            if "grad_clip" in cfg:
                torch.nn.utils.clip_grad_norm_(model.D.parameters(), cfg["grad_clip"])
            optimizer_d.step()
            
            time2 = time.time()
            
            bs = img.size(0)
            total_num += bs
            total_loss_g += loss_g * bs
            total_loss_d += loss_d * bs
            
            data_time += time1 - time0
            model_time += time2 - time1
            
            if (i % 50 == 0) and (rank == 0):
                logger.info('Iters: {} / {}, data time: {:.2f}, model time: {:.2f}, loss g: {:.4f} loss d: {:.4f}'.format(
                    i, len(trainloader), 
                    data_time / (i + 1), model_time / (i + 1),
                    total_loss_g.item() / total_num, total_loss_d.item() / total_num))
            
            if "scheduler" in cfg and not cfg["scheduler"]["by_epoch"]:
                scheduler_g.step()
                scheduler_d.step()
            
            time0 = time.time()

        if "scheduler" in cfg and cfg["scheduler"]["by_epoch"]:
            scheduler_g.step()
            scheduler_d.step()
        
        if args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0:
            IS, IS_std, FID = evaluate(model, cfg, sample_num=1000, local_rank=rank, word_size=word_size)
            if rank == 0:
                logger.info('***** Evaluation ***** >>>> IS: {:.2f} IS std : {:.2f} FID : {:.2f}\n'.format(
                    IS, IS_std, FID))
        
        # visualization
        with torch.no_grad():
            model.eval()
            sampled_imgs = model(num_samples=64)
            sampled_imgs = sampled_imgs * 0.5 + 0.5
        
        if rank == 0:
            torch.save(model.module.state_dict(),
                       os.path.join(args.save_path, 'latest.pth'))
            # save visualization
            save_image(sampled_imgs, os.path.join(args.save_path, f"visualization.jpg"), nrow=8)


if __name__ == '__main__':
    main()
