_base_ = [
    '../_base_/models/cascade_mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_tissuenet_n_instance.py',
    '../_base_/schedules/schedule_500e.py', '../_base_/default_runtime_interval25.py'
]
