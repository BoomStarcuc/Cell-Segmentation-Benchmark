_base_ = '../cascade_rcnn/cascade_mask_rcnn_r50_fpn_500e_coco_tissuenet_w.py'
model = dict(
    backbone=dict(
        type='Res2Net',
        depth=101,
        scales=4,
        base_width=26,
        in_channels=2,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://res2net101_v1d_26w_4s')))

evaluation = dict(interval=25, metric=['bbox','segm'])
