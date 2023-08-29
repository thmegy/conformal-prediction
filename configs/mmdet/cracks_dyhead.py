dataset_type = 'CocoDataset'
data_root = '/home/theo/workdir/mmdet/data/cracks_12_classes/'
backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=[(1000, 480), (1000, 600)],
        keep_ratio=True,
        backend='pillow'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(1000, 600), keep_ratio=True, backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type='CocoDataset',
            data_root='/home/theo/workdir/mmdet/data/cracks_12_classes/',
            ann_file='cracks_train.json',
            data_prefix=dict(
                img='/home/finn/DATASET/ai4cracks-dataset/images/train/'),
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            pipeline=[
                dict(type='LoadImageFromFile', backend_args=None),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(
                    type='RandomResize',
                    scale=[(1000, 480), (1000, 600)],
                    keep_ratio=True,
                    backend='pillow'),
                dict(type='RandomFlip', prob=0.5),
                dict(type='PackDetInputs')
            ],
            backend_args=None,
            metainfo=dict(
                classes=('Arrachement_pelade', 'Faiencage', 'Nid_de_poule',
                         'Transversale', 'Longitudinale',
                         'Pontage_de_fissures', 'Remblaiement_de_tranchees',
                         'Raccord_de_chaussee',
                         'Comblage_de_trou_ou_Projection_d_enrobe',
                         'Bouche_a_clef', 'Grille_avaloir',
                         'Regard_tampon')))))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='/home/theo/workdir/mmdet/data/cracks_12_classes/',
        ann_file='cracks_val.json',
        data_prefix=dict(
            img='/home/finn/DATASET/ai4cracks-dataset/images/val/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(
                type='Resize',
                scale=(1000, 600),
                keep_ratio=True,
                backend='pillow'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ],
        backend_args=None,
        metainfo=dict(
            classes=('Arrachement_pelade', 'Faiencage', 'Nid_de_poule',
                     'Transversale', 'Longitudinale', 'Pontage_de_fissures',
                     'Remblaiement_de_tranchees', 'Raccord_de_chaussee',
                     'Comblage_de_trou_ou_Projection_d_enrobe',
                     'Bouche_a_clef', 'Grille_avaloir', 'Regard_tampon'))))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='/home/theo/workdir/mmdet/data/cracks_12_classes/',
        ann_file='cracks_test.json',
        data_prefix=dict(
            img='/home/finn/DATASET/ai4cracks-dataset/images/test/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(
                type='Resize',
                scale=(1000, 600),
                keep_ratio=True,
                backend='pillow'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ],
        backend_args=None,
        metainfo=dict(
            classes=('Arrachement_pelade', 'Faiencage', 'Nid_de_poule',
                     'Transversale', 'Longitudinale', 'Pontage_de_fissures',
                     'Remblaiement_de_tranchees', 'Raccord_de_chaussee',
                     'Comblage_de_trou_ou_Projection_d_enrobe',
                     'Bouche_a_clef', 'Grille_avaloir', 'Regard_tampon'))))
val_evaluator = dict(
    type='CocoMetric',
    ann_file='/home/theo/workdir/mmdet/data/cracks_12_classes/cracks_val.json',
    metric=['bbox'],
    format_only=False,
    backend_args=None,
    classwise=True)
test_evaluator = dict(
    type='CocoMetric',
    ann_file='/home/theo/workdir/mmdet/data/cracks_12_classes/cracks_test.json',
    metric=['bbox'],
    format_only=False,
    backend_args=None,
    classwise=True)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=5e-05, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))),
    clip_grad=None)
auto_scale_lr = dict(enable=False, base_batch_size=16)
default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5000),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = 'outputs/cracks_dyhead_swin_20230717/epoch_9.pth'
resume = False
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'
model = dict(
    type='ATSS',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=128),
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'
        )),
    neck=[
        dict(
            type='FPN',
            in_channels=[384, 768, 1536],
            out_channels=256,
            start_level=0,
            add_extra_convs='on_output',
            num_outs=5),
        dict(
            type='DyHead',
            in_channels=256,
            out_channels=256,
            num_blocks=6,
            zero_init_offset=False)
    ],
    bbox_head=dict(
        type='ATSSHead',
        num_classes=12,
        in_channels=256,
        pred_kernel_size=1,
        stacked_convs=0,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128],
            center_offset=0.5),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
classes = ('Arrachement_pelade', 'Faiencage', 'Nid_de_poule', 'Transversale',
           'Longitudinale', 'Pontage_de_fissures', 'Remblaiement_de_tranchees',
           'Raccord_de_chaussee', 'Comblage_de_trou_ou_Projection_d_enrobe',
           'Bouche_a_clef', 'Grille_avaloir', 'Regard_tampon')
launcher = 'none'
work_dir = 'outputs/cracks_dyhead_swin_20230717/'
