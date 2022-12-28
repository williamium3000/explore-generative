model=dict(
    name="ddpm",
    backbone=dict(
        name="unet",
        kwargs=dict(
            T=1000,
            ch=128,
            ch_mult=[1, 2, 3, 4],
            attn=[2],
            num_res_blocks=2,
            dropout=0.15
        )
    ),
    beta_1=1e-4,
    beta_T=0.02,
    T=1000,
    img_shape=[3, 64, 64]
)

scheduler=dict(
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
batch_size = 32 # 4x

fid_statistics="utils/fid_statistics/fid_stats_celeba.npz"