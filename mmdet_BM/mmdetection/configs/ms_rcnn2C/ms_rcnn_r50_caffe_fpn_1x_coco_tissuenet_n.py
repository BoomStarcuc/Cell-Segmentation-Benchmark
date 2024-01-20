_base_ = '../mask_rcnn/mask_rcnn_r50_caffe_fpn_1x_coco_tissuenet_n.py'
model = dict(
    type='MaskScoringRCNN',
    backbone=dict(in_channels=2),
    roi_head=dict(
        type='MaskScoringRoIHead',
        mask_iou_head=dict(
            type='MaskIoUHead',
            num_convs=4,
            num_fcs=2,
            roi_feat_size=14,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            num_classes=1)),
    # model training and testing settings
    train_cfg=dict(rcnn=dict(mask_thr_binary=0.5)))

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[100, 150])
runner = dict(type='EpochBasedRunner', max_epochs=200)
evaluation = dict(interval=20, metric='segm')





