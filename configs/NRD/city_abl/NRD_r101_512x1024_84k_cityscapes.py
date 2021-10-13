_base_ = [
    '../../_base_/models/deeppad_r50.py',
    '../../_base_/datasets/cityscapes.py', '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_80k_warmup.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(
        norm_cfg=norm_cfg,
        depth=101,
    ),
    decode_head=dict(
        type='BilinearPADHead_fast',
        upsample_factor=8,
        dyn_branch_ch=16,
        mask_head_ch=16,
        norm_cfg=norm_cfg,
    ),
    auxiliary_head=dict(norm_cfg=norm_cfg)
)
