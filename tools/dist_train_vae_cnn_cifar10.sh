now=$(date +"%Y%m%d_%H%M%S")
save_path=work_dirs/vae_cnn_cifar10_norm
port=15872

if [ ! -d $save_path  ];then
  mkdir -p $save_path
fi

CUDA_VISIBLE_DEVICES=5 python \
    -m torch.distributed.launch --nproc_per_node=1 --master_port $port \
    train.py \
    --dataset cifar10 \
    --data ../dataSet/cifar10 \
    --eval_freq 10 \
    --cfg configs/vae_cnn_cifar10.py \
    --save-path $save_path --port $port 2>&1 | tee $save_path/$now.txt