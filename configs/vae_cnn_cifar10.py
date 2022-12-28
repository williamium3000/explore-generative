model = dict(
    name="vae_cnn",
    in_channels=3,
    latent_dim=512,
    hidden_dims=[64, 128, 256, 512],
    img_shape=[3, 32, 32],
    in_size=32,
    loss="mse_kl_loss_norm"
)
scheduler = dict(
    name="cosineannealinglr",
    by_epoch=True,
    kwargs=dict(
        eta_min=0
    )
)

# optim
optimizer="AdamW"
lr=1e-4
weight_decay=1e-4

epochs=200
batch_size=128 # 2x
fid_statistics="utils/fid_statistics/fid_stats_cifar10_train.npz"