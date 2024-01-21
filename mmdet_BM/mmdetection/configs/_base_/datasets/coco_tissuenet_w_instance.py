# dataset settings
dataset_type = 'CocoTissuenetDataset'
data_root = 'path/to/your/data/dir'

# [55.0704424 55.0704424 55.0704424]
# [67.22934807 67.22934807 67.22934807]
img_norm_cfg = dict(
    mean=[55.0704424, 55.0704424, 55.0704424], std=[67.22934807, 67.22934807, 67.22934807], to_rgb=True)
train_pipeline = [
    dict(type='LoadTIFImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(256, 256), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadTIFImageFromFile'),
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
data = dict(
    samples_per_gpu=24,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'tissuenet_wholecell_all_train.json',
        img_prefix=data_root +'wholecell/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'tissuenet_wholecell_all_val.json',
        img_prefix=data_root + 'wholecell/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'tissuenet_wholecell_all_test.json',
        img_prefix=data_root + 'wholecell/test',
        pipeline=test_pipeline))
evaluation = dict(interval=25, metric=['bbox'])
