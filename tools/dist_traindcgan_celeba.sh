now=$(date +"%Y%m%d_%H%M%S")
save_path=work_dirs/dcgan_celeba
port=15874

if [ ! -d $save_path  ];then
  mkdir -p $save_path
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 python \
    -m torch.distributed.launch --nproc_per_node=4 --master_port $port \
    train_gan.py \
    --dataset celeba \
    --data ../dataSet/celeba \
    --eval_freq 10 \
    --cfg configs/dcgan_celeba.py \
    --save-path $save_path --port $port 2>&1 | tee $save_path/$now.txt