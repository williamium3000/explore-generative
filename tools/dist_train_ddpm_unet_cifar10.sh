now=$(date +"%Y%m%d_%H%M%S")
save_path=work_dirs/ddpm_unet_cifar10
port=15869

if [ ! -d $save_path  ];then
  mkdir -p $save_path
fi

CUDA_VISIBLE_DEVICES=0,1 python \
    -m torch.distributed.launch --nproc_per_node=2 --master_port $port \
    train.py \
    --dataset cifar10 \
    --data ../dataSet/cifar10 \
    --eval_freq 10 \
    --cfg configs/ddpm_unet_cifar10.py \
    --save-path $save_path --port $port 2>&1 | tee $save_path/$now.txt