model=dict(
    name="ddpm",
    backbone=dict(
        name="unet",
        T=1000,
        ch=128,
        ch_mult=[1, 2, 3, 4],
        attn=[2],
        num_res_blocks=2,
        dropout=0.15
    ),
    beta_1=1e-4,
    beta_T=0.02,
    img_shape=[3, 32, 32]
)
pretrained=True

num_classes = 1000
criterion = dict(
    name="CELoss"
)
scheduler = dict(
    name="StepLR",
    by_epoch=True,
    kwargs=dict(
        step_size=30, gamma=0.1
    )
)


# optim
optimizer="AdamW"
lr=1e-4
weight_decay=1e-4

epochs=200
batch_size = 64 # 2x
