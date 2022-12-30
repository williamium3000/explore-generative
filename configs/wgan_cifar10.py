model = dict(
    name="wgan",
    latent_dim=128,
    img_shape=[3, 64, 64]
)

# optim
optimizer="RMSprop"
lr=0.00005
weight_decay=1e-4
momentum=0.0

# wgan
lipschitz_clip=0.01
n_critic=5

epochs=200
batch_size=64 # 2x
fid_statistics="utils/fid_statistics/fid_stats_cifar10_train.npz"
