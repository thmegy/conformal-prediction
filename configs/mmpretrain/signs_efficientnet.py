model = dict(
    type='ImageClassifier',
    backbone=dict(type='EfficientNet', arch='b4'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=91,
        in_channels=1792,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)))
dataset_type = 'ImageNet'
data_preprocessor = dict(
    num_classes=1000,
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='EfficientNetRandomCrop', scale=380),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='EfficientNetCenterCrop', crop_size=380),
    dict(type='PackInputs')
]
train_dataloader = dict(
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    batch_size=12,
    num_workers=5,
    dataset=dict(
        type='CustomDataset',
        data_prefix='/home/theo/workdir/mmcls/data/signs/train/',
        with_label=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='EfficientNetRandomCrop', scale=380),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='PackInputs')
        ]),
    sampler=dict(type='DefaultSampler', shuffle=True))
val_dataloader = dict(
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    batch_size=12,
    num_workers=5,
    dataset=dict(
        type='CustomDataset',
        data_prefix='/home/theo/workdir/mmcls/data/signs/val/',
        with_label=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='EfficientNetCenterCrop', crop_size=380),
            dict(type='PackInputs')
        ]),
    sampler=dict(type='DefaultSampler', shuffle=False))
val_evaluator = dict(type='Accuracy', topk=(1, 5))
test_dataloader = dict(
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    batch_size=12,
    num_workers=5,
    dataset=dict(
        type='CustomDataset',
        data_prefix='/home/theo/workdir/mmcls/data/signs/test/',
        with_label=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='EfficientNetCenterCrop', crop_size=380),
            dict(type='PackInputs')
        ]),
    sampler=dict(type='DefaultSampler', shuffle=False))
test_evaluator = dict(type='Accuracy', topk=(1, 5))
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001))
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[30, 60, 90], gamma=0.1)
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(base_batch_size=256, enable=True)
default_scope = 'mmpretrain'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=500),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=2),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='UniversalVisualizer', vis_backends=[dict(type='LocalVisBackend')])
log_level = 'INFO'
load_from = 'pretrained_checkpoints/efficientnet-b4_3rdparty_8xb32-aa-advprop_in1k_20220119-38c2238c.pth'
resume = False
randomness = dict(seed=None, deterministic=False)
launcher = 'none'
work_dir = 'output/signs_efficientnet_20230731'
