_base_ = [
    '../../_base_/models/deeppad_r50.py',
    '../../_base_/datasets/ade20k.py', '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_warmup.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    pretrained='/pretrained/alt_gvt_large.pth',
    backbone=dict(
        type='alt_gvt_large',
        style='pytorch',
        norm_cfg=norm_cfg,
    ),
    decode_head=dict(
        type='BilinearPADHead_fast',
        num_classes=150,
        upsample_factor=8,
        dyn_branch_ch=16,
        mask_head_ch=16,
        in_channels=1024,
        c1_in_channels=128,
        norm_cfg=norm_cfg,
    ),
    auxiliary_head=dict(
        norm_cfg=norm_cfg,
        in_channels=512,
        num_classes=150)
)

optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)