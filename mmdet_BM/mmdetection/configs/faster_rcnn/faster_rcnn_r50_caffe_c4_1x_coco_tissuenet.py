_base_ = [
    '../_base_/models/faster_rcnn_r50_caffe_c4.py',
    '../_base_/datasets/coco_tissuenet_detection.py',
    '../_base_/schedules/schedule_3x.py', '../_base_/default_runtime.py'
]
# use caffe img_norm
img_norm_cfg = dict(
    mean=[55.0704424, 55.0704424, 55.0704424], std=[67.22934807, 67.22934807, 67.22934807], to_rgb=False)
train_pipeline = [
    dict(type='LoadTIFImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(256, 256), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
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
dataset_type = 'CocoTissuenetDataset'
data_root = '/home/hx5239/data/val_merge/'
data = dict(
    _delete_=True,
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'tissuenet_nuclear_all_train.json',
        img_prefix=data_root +'coco_to_mmseg/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'tissuenet_nuclear_all_val.json',
        img_prefix=data_root + 'coco_to_mmseg/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'tissuenet_nuclear_all_test.json',
        img_prefix=data_root + 'coco_to_mmseg/test',
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'])
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
