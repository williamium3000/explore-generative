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

# optim
optimizer="AdamW"
lr=5e-4
weight_decay=1e-4

epochs=200
batch_size = 256 # 2x
