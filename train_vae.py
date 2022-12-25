import argparse
import logging
import os
import pprint
import time
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

from utils.utils import count_params, init_log, accuracy
from utils.dist import setup_distributed


from models.vae.vae import VAE

parser = argparse.ArgumentParser(description='train domain generalization (oracle)')
parser.add_argument('--cfg', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def evaluate(model, loader, cfg, criterion, local_rank=-1):
    model.eval()
    total_loss = 0.0
    total_acc1 = 0.0
    total_acc5 = 0.0
    total_num = 0.0
    with torch.no_grad():
        for image, target in tqdm.tqdm(loader, disable=local_rank != 0 and local_rank != -1):
            image = image.cuda()
            target = target.cuda()
            output = model(image)
            loss = criterion(output, target)
            
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            bs = image.size(0)
            total_loss += loss * bs
            total_acc1 += acc1 * bs
            total_acc5 += acc5 * bs
            total_loss += loss * bs
            total_num += bs

    dist.all_reduce(total_acc1)
    dist.all_reduce(total_acc5)
    dist.all_reduce(total_loss)
    
    total_num = torch.tensor(total_num).cuda()
    dist.all_reduce(total_num)

    return total_acc1 / total_num, total_acc5 / total_num, total_loss / total_num

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

    model = VAE(768, 256)
    
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

    if cfg['criterion']['name'] == 'CELoss':
        criterion = nn.CrossEntropyLoss().cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])
    
    trainset = torchvision.datasets.CIFAR10(
        root='../dataSet/cifar10/', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'],
                             pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler)

    if "scheduler" in cfg:
        if cfg["scheduler"]["name"] == "StepLR":
            scheduler = lr_scheduler.StepLR(optimizer, **cfg["scheduler"]["kwargs"])
        elif args.lr_scheduler == "cosineannealinglr":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg['epochs'] * len(trainloader), **cfg["scheduler"]["kwargs"]
            )
        elif args.lr_scheduler == "exponentiallr":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **cfg["scheduler"]["kwargs"])

    for epoch in range(cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.4f}'.format(
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
            loss = model(img)

            optimizer.zero_grad()
            loss.backward()
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
        
        # top1_acc_val, top5_acc_val, val_loss_val = evaluate(model, valloader, cfg, criterion, local_rank=rank)
        # top1_acc_test, top5_acc_test, val_loss_test = evaluate(model, testloader, cfg, criterion, local_rank=rank)

        if rank == 0:
            logger.info('***** Evaluation ***** >>>>')
            # logger.info('Val:  Top1: {:.2f} Top5: {:.2f} Loss: {:.4f}\n'.format(
            #     top1_acc_val.item() * 100, top5_acc_val.item() * 100, val_loss_val.item()))
            # logger.info('Test:  Top1: {:.2f} Top5: {:.2f} Loss: {:.4f}\n'.format(
            #     top1_acc_test.item() * 100, top5_acc_test.item() * 100, val_loss_test.item()))

        # if top1_acc_val > previous_best and rank == 0:
        #     if previous_best != 0:
        #         os.remove(os.path.join(args.save_path, 'best_val{:.2f}_test{:.2f}.pth'.format(100 * previous_best, 100 * according_test)))
        #     previous_best = top1_acc_val
        #     according_test = top1_acc_test
        #     torch.save(model.module.state_dict(),
        #                os.path.join(args.save_path, 'best_val{:.2f}_test{:.2f}.pth'.format(100 * previous_best, 100 * according_test)))


if __name__ == '__main__':
    main()
