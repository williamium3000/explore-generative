now=$(date +"%Y%m%d_%H%M%S")
save_path=work_dirs/vae_mlp_cifar10
port="15865"

CUDA_VISIBLE_DEVICES=4,5,6,7 python \
    -m torch.distributed.launch --nproc_per_node=4 --master_port $port \
    train_vae.py \
     --cfg configs/vae_mlp_cifar10.py \
    --save-path $save_path --port $port 2>&1 | tee $save_path/$now.txt