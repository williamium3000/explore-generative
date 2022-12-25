model = "resnet50"
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
optimizer="SGD"
lr=0.001
momentum=0.9
weight_decay=1e-4

epochs=90
batch_size = 64 # 4x
