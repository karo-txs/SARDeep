runner = dict(type='EpochBasedRunner', max_epochs=50)
evaluation = dict(interval=1, metric=['mAP'])

checkpoint_config = dict(interval=5)
seed = 0
# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [
    dict(type='NumClassCheckHook'),
]

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, by_epoch=True, save_best='acc', rule='less'))

dist_params = dict(backend='nccl')
log_level = 'INFO'

load_from = f"checkpoints/fine_tune.pth"
work_dir = f"outputs/model_file"

resume_from = None
workflow = [('train', 1)]

gpu_ids = range(1)
device = 'cpu'

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
