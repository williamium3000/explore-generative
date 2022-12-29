model = dict(
    name="dcgan",
    latent_dim=128,
    img_shape=[3, 64, 64]
)
scheduler = dict(
    name="cosineannealinglr",
    by_epoch=True,
    kwargs=dict(
        eta_min=0
    )
)

# optim
optimizer="Adam"
lr=2e-4
weight_decay=1e-4
betas=(0.5, 0.999)

epochs=200
batch_size=64 # 2x
fid_statistics="utils/fid_statistics/fid_stats_cifar10_train.npz"