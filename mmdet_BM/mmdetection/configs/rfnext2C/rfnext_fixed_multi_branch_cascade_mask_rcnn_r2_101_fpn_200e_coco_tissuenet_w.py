_base_ = '../res2net2C/cascade_mask_rcnn_r2_101_fpn_500e_coco_tissuenet_w.py'

custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(
        type='RFSearchHook',
        mode='fixed_multi_branch',
        rfstructure_file=  # noqa
        'mmdetection/configs/rfnext2C/search_log/cascade_mask_rcnn_r2_101_fpn_20e_coco/local_search_config_step11.json',  # noqa
        verbose=True,
        by_epoch=True,
        config=dict(
            search=dict(
                step=0,
                max_step=12,
                search_interval=1,
                exp_rate=0.5,
                init_alphas=0.01,
                mmin=1,
                mmax=24,
                num_branches=2,
                skip_layer=['stem', 'layer1'])))
]

img_norm_cfg = dict(
    mean=[55.0704424, 55.0704424], std=[67.22934807, 67.22934807], to_rgb=False)
train_pipeline = [
    dict(type='LoadTIFImageFromFile2C'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(256, 256), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadTIFImageFromFile2C'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 256),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

dataset_type = 'CocoTissuenetDataset'
data_root = 'path/to/your/data/dir'
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'tissuenet_wholecell_all_train_2C.json',
        img_prefix=data_root +'train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'tissuenet_wholecell_all_val_2C.json',
        img_prefix=data_root +'val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'tissuenet_wholecell_all_test_2C.json',
        img_prefix=data_root +'test',
        pipeline=test_pipeline))
evaluation = dict(interval=25, metric=['bbox','segm'])

optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[150, 180])
runner = dict(type='EpochBasedRunner', max_epochs=200)
