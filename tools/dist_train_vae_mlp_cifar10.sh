now=$(date +"%Y%m%d_%H%M%S")
save_path=work_dirs/vae_mlp2_cifar10
port=15866

if [ ! -d $save_path  ];then
  mkdir -p $save_path
fi

CUDA_VISIBLE_DEVICES=4,5 python \
    -m torch.distributed.launch --nproc_per_node=2 --master_port $port \
    train.py \
     --cfg configs/vae_mlp_cifar10.py \
    --save-path $save_path --port $port 2>&1 | tee $save_path/$now.txt