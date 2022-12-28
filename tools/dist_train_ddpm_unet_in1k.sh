now=$(date +"%Y%m%d_%H%M%S")
save_path=work_dirs/ddpm_unet_imagenet
port=15870

if [ ! -d $save_path  ];then
  mkdir -p $save_path
fi

CUDA_VISIBLE_DEVICES=4,5,6,7 python \
    -m torch.distributed.launch --nproc_per_node=4 --master_port $port \
    train.py \
    --dataset imagenet \
    --data data/imagenet \
    --eval_freq 1 \
    --cfg configs/ddpm_unet_in1k.py \
    --save-path $save_path --port $port 2>&1 | tee $save_path/$now.txt