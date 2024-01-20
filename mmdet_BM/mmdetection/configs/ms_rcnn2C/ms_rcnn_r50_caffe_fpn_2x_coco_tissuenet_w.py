_base_ = './ms_rcnn_r50_caffe_fpn_1x_coco_tissuenet_w.py'

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
    _delete_=True,
    samples_per_gpu=36,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'tissuenet_wholecell_all_train_2C.json',
        img_prefix=data_root +'train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'tissuenet_wholecell_all_val_2C.json',
        img_prefix=data_root + 'val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'tissuenet_wholecell_all_test_2C.json',
        img_prefix=data_root + 'test',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric=['bbox'])

# learning policy
lr_config = dict(step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=50)
