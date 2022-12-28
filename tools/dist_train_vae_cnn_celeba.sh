now=$(date +"%Y%m%d_%H%M%S")
save_path=work_dirs/vae_cnn_celeba_norm
port=15875

if [ ! -d $save_path  ];then
  mkdir -p $save_path
fi

CUDA_VISIBLE_DEVICES=6,7 python \
    -m torch.distributed.launch --nproc_per_node=2 --master_port $port \
    train.py \
    --dataset celeba \
    --data ../dataSet/celeba \
    --eval_freq 10 \
    --cfg configs/vae_cnn_celeba.py \
    --save-path $save_path --port $port 2>&1 | tee $save_path/$now.txt