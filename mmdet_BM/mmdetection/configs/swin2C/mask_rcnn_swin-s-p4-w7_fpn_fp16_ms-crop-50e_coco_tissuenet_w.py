_base_ = './mask_rcnn_swin-t-p4-w7_fpn_ms-crop-50e_coco_tissuenet_w.py'
# pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'  # noqa
model = dict(
    backbone=dict(
        in_channels=2,
        depths=[2, 2, 18, 2]))

img_norm_cfg = dict(
    mean=[123.675, 116.28], std=[58.395, 57.12], to_rgb=False)
# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadTIFImageFromFile2C'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[
            dict(
                type='Resize',
                img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                           (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                           (736, 1333), (768, 1333), (800, 1333)],
                multiscale_mode='value',
                keep_ratio=True)
        ],
                  [
                      dict(
                          type='Resize',
                          img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                          multiscale_mode='value',
                          keep_ratio=True),
                      dict(
                          type='RandomCrop',
                          crop_type='absolute_range',
                          crop_size=(384, 600),
                          allow_negative_crop=True),
                      dict(
                          type='Resize',
                          img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                     (576, 1333), (608, 1333), (640, 1333),
                                     (672, 1333), (704, 1333), (736, 1333),
                                     (768, 1333), (800, 1333)],
                          multiscale_mode='value',
                          override=True,
                          keep_ratio=True)
                  ]]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

test_pipeline = [
    dict(type='LoadTIFImageFromFile2C'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(480, 480), (512, 512), (544, 544),
                    (576, 576), (608, 608), (640, 640),
                    (672, 672), (704, 704), (736, 736),
                    (768, 768), (800, 800)],
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
# evaluation = dict(interval=5, metric=['bbox','segm'])
evaluation = dict(interval=25, metric=['bbox'])