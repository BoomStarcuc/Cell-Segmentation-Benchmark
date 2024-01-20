_base_ = './htc_r50_fpn_1x_coco_tissuenet_n.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
# learning policy
lr_config = dict(step=[100, 150])
runner = dict(type='EpochBasedRunner', max_epochs=200)
