_base_ = [
    '../../_base_/models/deeppad_r50.py',
    '../../_base_/datasets/pascal_context_59.py', '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_80k_warmup.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(
        depth=101,
        norm_cfg=norm_cfg,
    ),
    decode_head=dict(
        type='BilinearPADHead_fast',
        num_classes=59,
        upsample_factor=8,
        dyn_branch_ch=16,
        mask_head_ch=16,
        norm_cfg=norm_cfg,
    ),
    auxiliary_head=dict(
        norm_cfg=norm_cfg,
        num_classes=59),
    test_cfg=dict(mode='whole')
)
optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001)
