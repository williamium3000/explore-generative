model = dict(
    name="vae_mlp",
    input_size=3 * 32 * 32,
    latent_dim=512,
    img_shape=[3, 32, 32],
    loss="mse_kl_loss"
)
scheduler = dict(
    name="cosineannealinglr",
    by_epoch=True,
    kwargs=dict(
        eta_min=0
    )
)

# optim: sgd
optimizer="SGD"
lr=1e-3
momentum=0.9
weight_decay=1e-4

# optim: AdamW
# optimizer="AdamW"
# lr=1e-5
# weight_decay=1e-4

epochs=200
batch_size=128 # 2x
fid_statistics="utils/fid_statistics/fid_stats_cifar10_train.npz"