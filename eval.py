import argparse
import os
import pprint
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn

import tqdm
from mmcv import Config
from torchvision.utils import save_image
from utils.utils import count_params, FolderDataset
from utils.dist import setup_distributed
from utils.inception_score import inception_score
from pytorch_fid.fid_score import calculate_fid_given_paths
from models.builder import build_model

parser = argparse.ArgumentParser(description='train domain generalization (oracle)')
parser.add_argument('--cfg', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True, choices=["cifar10", "cifar100", "celeba", "imagenet", "mnist"])
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--ckpt', type=int, required=True)
parser.add_argument('--save-path', type=str, required=True)

def evaluate(model, cfg, sample_num, local_rank, word_size):
    with TemporaryDirectory() as temp_dir:
        # broadcast temp_dir to make sure they are the same on each device
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if local_rank == 0:
            temp_dir = torch.tensor(
                bytearray(temp_dir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(temp_dir)] = temp_dir
        dist.broadcast(dir_tensor, 0)
        temp_dir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
        
        print(f"saving in temp dir {temp_dir}.")
        sample(model, cfg, sample_num, temp_dir, local_rank, word_size)
        
        IS, IS_std, FID = 0.0, 0.0, 0.0
        if local_rank == 0:
            temp_dataset = FolderDataset(temp_dir, transform=transforms.Compose([transforms.Resize(32),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                        ]))
            IS, IS_std = inception_score(temp_dataset, batch_size=64, resize=True, splits=10)
            FID = calculate_fid_given_paths(
                paths=[temp_dir, cfg["fid_statistics"]],
                batch_size=64, device='cuda',
                dims=2048, num_workers=4)
        dist.barrier()
        # sync metrics
        IS = torch.ones(1, 1).cuda() * IS
        IS_std = torch.ones(1, 1).cuda() * IS_std
        FID = torch.ones(1, 1).cuda() * FID
        dist.broadcast(IS, 0)
        dist.broadcast(IS_std, 0)
        dist.broadcast(FID, 0)
        
        return IS.item(), IS_std.item(), FID.item()


def sample(model, cfg, sample_num, save_path, local_rank, word_size):
    model.eval()
    with torch.no_grad():
        for i in tqdm.tqdm(range(local_rank, sample_num // cfg['batch_size'], word_size)):
            sampled_imgs = model(num_samples=cfg['batch_size'])
            sampled_imgs = sampled_imgs * 0.5 + 0.5
            # save each individual image
            for j in range(cfg['batch_size']):
                save_image(sampled_imgs[j].unsqueeze(0), os.path.join(save_path, f"{i * cfg['batch_size'] + j}.jpg"))
        # sample the rest with rank 0
        if local_rank == 0:
            sampled_imgs = model(num_samples=sample_num - (sample_num // cfg['batch_size']) * cfg['batch_size'])
            sampled_imgs = sampled_imgs * 0.5 + 0.5
            # save each individual image
            for j in range(sampled_imgs.size(0)):
                save_image(sampled_imgs[j].unsqueeze(0), os.path.join(save_path, f"{i * cfg['batch_size'] + j}.jpg"))
               
def main():
    args = parser.parse_args()
    cfg = Config.fromfile(args.cfg)
    

    rank, word_size = setup_distributed(port=args.port)

    if rank == 0:
        print('{}\n'.format(pprint.pformat(cfg)))

    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    # model = VAE(in_channels=3, latent_dim=512, hidden_dims=[32, 128, 512])
    model = build_model(cfg)
    
    if rank == 0:
        print('Total params: {:.1f}M\n'.format(count_params(model)))

    local_rank = int(os.environ["LOCAL_RANK"])
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=False)


    IS, IS_std, FID = evaluate(model, cfg, sample_num=1000, local_rank=rank, word_size=word_size)
    if rank == 0:
        print('***** Evaluation ***** >>>> IS: {:.2f} IS std : {:.2f} FID : {:.2f}\n'.format(
            IS, IS_std, FID))
        
    if rank == 0:
        # visualization
        with torch.no_grad():
            sampled_imgs = model(num_samples=64)
            sampled_imgs = sampled_imgs * 0.5 + 0.5
            save_image(sampled_imgs, os.path.join(args.save_path, f"visualization.jpg"), nrow=8)


if __name__ == '__main__':
    main()
