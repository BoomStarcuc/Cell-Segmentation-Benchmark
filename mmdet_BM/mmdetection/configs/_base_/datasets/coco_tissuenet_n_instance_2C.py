# dataset settings
dataset_type = 'CocoTissuenetDataset'
data_root = '/shared/rc/spl/hx5239_homedir/hx5239/data/val_merge/COCO_TissueNet_2Channel/'

# [55.0704424 55.0704424 55.0704424]
# [67.22934807 67.22934807 67.22934807]
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
data = dict(
    samples_per_gpu=12,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'tissuenet_nuclear_all_train_2C.json',
        img_prefix=data_root +'train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'tissuenet_nuclear_all_val_2C.json',
        img_prefix=data_root +'val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'tissuenet_nuclear_all_test_2C.json',
        img_prefix=data_root +'test',
        pipeline=test_pipeline))
# evaluation = dict(interval=5, metric=['bbox','segm'])
evaluation = dict(interval=25, metric=['bbox'])
