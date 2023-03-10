now=$(date +"%Y%m%d_%H%M%S")
save_path=work_dirs/dcgan_cifar10_newcode
port=15885

if [ ! -d $save_path  ];then
  mkdir -p $save_path
fi

CUDA_VISIBLE_DEVICES=6,7 python \
    -m torch.distributed.launch --nproc_per_node=2 --master_port $port \
    train_gan.py \
    --dataset cifar10 \
    --data ../dataSet/cifar10 \
    --eval_freq 10 \
    --cfg configs/dcgan_cifar10.py \
    --save-path $save_path --port $port 2>&1 | tee $save_path/$now.txt