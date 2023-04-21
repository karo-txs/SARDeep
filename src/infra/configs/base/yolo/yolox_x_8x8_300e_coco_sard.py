_base_ = [
    '../_base_/runtime_v1.py',
    '../_base_/schedule_3x.py',
    '../_base_/sard_dataset.py',
]
num_classes = 1
img_scale = (1333, 800)

# model settings
model = dict(
    type='YOLOX',
    input_size=img_scale,
    random_size_range=(15, 25),
    random_size_interval=10,
    backbone=dict(deepen_factor=1.33, widen_factor=1.25),
    neck=dict(in_channels=[320, 640, 1280], out_channels=320, num_csp_blocks=4),
    bbox_head=dict(type='YOLOXHead', num_classes=num_classes, in_channels=128, feat_channels=320),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))