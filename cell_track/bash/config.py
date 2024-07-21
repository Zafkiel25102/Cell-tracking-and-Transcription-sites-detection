_backend_args = None
_multiscale_resize_transforms = [
    dict(
        transforms=[
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
    dict(
        transforms=[
            dict(scale=(
                320,
                320,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    320,
                    320,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
    dict(
        transforms=[
            dict(scale=(
                960,
                960,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    960,
                    960,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
]
affine_scale = 0.9
albu_train_transforms = [
    dict(p=0.01, type='Blur'),
    dict(p=0.01, type='MedianBlur'),
    dict(p=0.01, type='ToGray'),
    dict(p=0.01, type='CLAHE'),
]
anchors = [
    [
        (
            68,
            69,
        ),
        (
            154,
            91,
        ),
        (
            143,
            162,
        ),
    ],
    [
        (
            242,
            160,
        ),
        (
            189,
            287,
        ),
        (
            391,
            207,
        ),
    ],
    [
        (
            353,
            337,
        ),
        (
            539,
            341,
        ),
        (
            443,
            432,
        ),
    ],
]
backend_args = None
base_lr = 0.0025
batch_shapes_cfg = None
class_name = ('cell', )
close_mosaic_epochs = 10
copypaste_prob = 0.3
custom_hooks = [
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        priority=49,
        strict_load=False,
        type='EMAHook',
        update_buffers=True),
    dict(
        switch_epoch=490,
        switch_pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(
                mask2bbox=True,
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True),
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=True,
                pad_val=dict(img=114.0),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
            dict(
                border_val=(
                    114,
                    114,
                    114,
                ),
                max_aspect_ratio=100,
                max_rotate_degree=0.0,
                max_shear_degree=0.0,
                min_area_ratio=0.01,
                scaling_ratio_range=(
                    0.09999999999999998,
                    1.9,
                ),
                type='YOLOv5RandomAffine',
                use_mask_refine=True),
            dict(keys=[
                'gt_masks',
            ], type='RemoveDataElement'),
            dict(
                bbox_params=dict(
                    format='pascal_voc',
                    label_fields=[
                        'gt_bboxes_labels',
                        'gt_ignore_flags',
                    ],
                    type='BboxParams'),
                keymap=dict(gt_bboxes='bboxes', img='image'),
                transforms=[
                    dict(p=0.01, type='Blur'),
                    dict(p=0.01, type='MedianBlur'),
                    dict(p=0.01, type='ToGray'),
                    dict(p=0.01, type='CLAHE'),
                ],
                type='mmdet.Albu'),
            dict(type='YOLOv5HSVRandomAug'),
            dict(prob=0.5, type='mmdet.RandomFlip'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'flip',
                    'flip_direction',
                ),
                type='mmdet.PackDetInputs'),
        ],
        type='mmdet.PipelineSwitchHook'),
]
data_root = 'data/pothole_coco/'
dataset_type = 'YOLOv5CocoDataset'
deepen_factor = 1.0
default_hooks = dict(
    checkpoint=dict(
        interval=10, max_keep_ckpts=2, save_best='auto',
        type='CheckpointHook'),
    logger=dict(interval=5, type='LoggerHook'),
    param_scheduler=dict(
        lr_factor=0.01,
        max_epochs=40,
        scheduler_type='linear',
        type='YOLOv5ParamSchedulerHook',
        warmup_mim_iter=10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        draw=True,
        test_out_dir='show_results',
        type='mmdet.DetVisualizationHook'))
default_scope = 'mmyolo'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_scale = (
    640,
    640,
)
img_scales = [
    (
        640,
        640,
    ),
    (
        320,
        320,
    ),
    (
        960,
        960,
    ),
]
last_stage_out_channels = 512
last_transform = [
    dict(keys=[
        'gt_masks',
    ], type='RemoveDataElement'),
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', img='image'),
        transforms=[
            dict(p=0.01, type='Blur'),
            dict(p=0.01, type='MedianBlur'),
            dict(p=0.01, type='ToGray'),
            dict(p=0.01, type='CLAHE'),
        ],
        type='mmdet.Albu'),
    dict(type='YOLOv5HSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
        ),
        type='mmdet.PackDetInputs'),
]
launcher = 'none'
load_from = 'work_dirs/yolov8_x_mask-refine_syncbn_fast_8xb16-500e_cell/best_coco_bbox_mAP_epoch_30.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
loss_bbox_weight = 7.5
loss_cls_weight = 0.5
loss_dfl_weight = 0.375
lr_factor = 0.01
max_aspect_ratio = 100
max_epochs = 40
max_keep_ckpts = 2
metainfo = dict(
    classes=('cell', ), palette=[
        (
            20,
            220,
            60,
        ),
    ])
min_area_ratio = 0.01
mixup_prob = 0.15
model = dict(
    backbone=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        arch='P5',
        deepen_factor=1.0,
        frozen_stages=4,
        last_stage_out_channels=512,
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        type='YOLOv8CSPDarknet',
        widen_factor=1.25),
    bbox_head=dict(
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        head_module=dict(
            act_cfg=dict(inplace=True, type='SiLU'),
            featmap_strides=[
                8,
                16,
                32,
            ],
            in_channels=[
                256,
                512,
                512,
            ],
            norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
            num_classes=1,
            reg_max=16,
            type='YOLOv8HeadModule',
            widen_factor=1.25),
        loss_bbox=dict(
            bbox_format='xyxy',
            iou_mode='ciou',
            loss_weight=7.5,
            reduction='sum',
            return_iou=False,
            type='IoULoss'),
        loss_cls=dict(
            loss_weight=0.006250000000000001,
            reduction='none',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True),
        loss_dfl=dict(
            loss_weight=0.375,
            reduction='mean',
            type='mmdet.DistributionFocalLoss'),
        prior_generator=dict(
            offset=0.5, strides=[
                8,
                16,
                32,
            ], type='mmdet.MlvlPointGenerator'),
        type='YOLOv8Head'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            0.0,
            0.0,
            0.0,
        ],
        std=[
            255.0,
            255.0,
            255.0,
        ],
        type='YOLOv5DetDataPreprocessor'),
    neck=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        deepen_factor=1.0,
        in_channels=[
            256,
            512,
            512,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        num_csp_blocks=3,
        out_channels=[
            256,
            512,
            512,
        ],
        type='YOLOv8PAFPN',
        widen_factor=1.25),
    test_cfg=dict(
        max_per_img=300,
        multi_label=True,
        nms=dict(iou_threshold=0.3, type='nms'),
        nms_pre=30000,
        score_thr=0.001),
    train_cfg=dict(
        assigner=dict(
            alpha=0.5,
            beta=6.0,
            eps=1e-09,
            num_classes=1,
            topk=10,
            type='BatchTaskAlignedAssigner',
            use_ciou=True),
        initial_assigner=dict(num_classes=1)),
    type='YOLODetector')
model_test_cfg = dict(
    max_per_img=300,
    multi_label=True,
    nms=dict(iou_threshold=0.3, type='nms'),
    nms_pre=30000,
    score_thr=0.001)
mosaic_affine_transform = [
    dict(
        img_scale=(
            640,
            640,
        ),
        pad_val=114.0,
        pre_transform=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(
                mask2bbox=True,
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True),
        ],
        type='Mosaic'),
    dict(prob=0.3, type='YOLOv5CopyPaste'),
    dict(
        border=(
            -320,
            -320,
        ),
        border_val=(
            114,
            114,
            114,
        ),
        max_aspect_ratio=100.0,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        min_area_ratio=0.01,
        scaling_ratio_range=(
            0.09999999999999998,
            1.9,
        ),
        type='YOLOv5RandomAffine',
        use_mask_refine=True),
]
norm_cfg = dict(eps=0.001, momentum=0.03, type='BN')
num_classes = 1
num_det_layers = 3
optim_wrapper = dict(
    clip_grad=dict(max_norm=10.0),
    constructor='YOLOv5OptimizerConstructor',
    optimizer=dict(
        batch_size_per_gpu=16,
        lr=0.0025,
        momentum=0.937,
        nesterov=True,
        type='SGD',
        weight_decay=0.0005),
    type='OptimWrapper')
param_scheduler = None
persistent_workers = True
pre_transform = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        mask2bbox=True, type='LoadAnnotations', with_bbox=True,
        with_mask=True),
]
resume = False
save_epoch_intervals = 10
strides = [
    8,
    16,
    32,
]
tal_alpha = 0.5
tal_beta = 6.0
tal_topk = 10
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/instances_val2017.json',
        batch_shapes_cfg=None,
        data_prefix=dict(img='images/'),
        data_root='data/pothole_coco/',
        metainfo=dict(classes=('cell', ), palette=[
            (
                20,
                220,
                60,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='YOLOv5CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/pothole_coco/annotations/instances_val2017.json',
    metric='bbox',
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='mmdet.CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(scale=(
        640,
        640,
    ), type='YOLOv5KeepRatioResize'),
    dict(
        allow_scale_up=False,
        pad_val=dict(img=114),
        scale=(
            640,
            640,
        ),
        type='LetterResize'),
    dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'pad_param',
        ),
        type='mmdet.PackDetInputs'),
]
train_ann_file = 'data/pothole_coco/annotations/instances_train2017.json'
train_batch_size_per_gpu = 4
train_cfg = dict(
    dynamic_intervals=[
        (
            490,
            1,
        ),
    ],
    max_epochs=40,
    type='EpochBasedTrainLoop',
    val_interval=10)
train_data_prefix = 'train2017/'
train_dataloader = dict(
    batch_size=4,
    collate_fn=dict(type='yolov5_collate'),
    dataset=dict(
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='images/'),
        data_root='data/pothole_coco/',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        metainfo=dict(classes=('cell', ), palette=[
            (
                20,
                220,
                60,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(
                mask2bbox=True,
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True),
            dict(
                img_scale=(
                    640,
                    640,
                ),
                pad_val=114.0,
                pre_transform=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(
                        mask2bbox=True,
                        type='LoadAnnotations',
                        with_bbox=True,
                        with_mask=True),
                ],
                type='Mosaic'),
            dict(prob=0.3, type='YOLOv5CopyPaste'),
            dict(
                border=(
                    -320,
                    -320,
                ),
                border_val=(
                    114,
                    114,
                    114,
                ),
                max_aspect_ratio=100.0,
                max_rotate_degree=0.0,
                max_shear_degree=0.0,
                min_area_ratio=0.01,
                scaling_ratio_range=(
                    0.09999999999999998,
                    1.9,
                ),
                type='YOLOv5RandomAffine',
                use_mask_refine=True),
            dict(
                pre_transform=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(
                        mask2bbox=True,
                        type='LoadAnnotations',
                        with_bbox=True,
                        with_mask=True),
                    dict(
                        img_scale=(
                            640,
                            640,
                        ),
                        pad_val=114.0,
                        pre_transform=[
                            dict(backend_args=None, type='LoadImageFromFile'),
                            dict(
                                mask2bbox=True,
                                type='LoadAnnotations',
                                with_bbox=True,
                                with_mask=True),
                        ],
                        type='Mosaic'),
                    dict(prob=0.3, type='YOLOv5CopyPaste'),
                    dict(
                        border=(
                            -320,
                            -320,
                        ),
                        border_val=(
                            114,
                            114,
                            114,
                        ),
                        max_aspect_ratio=100.0,
                        max_rotate_degree=0.0,
                        max_shear_degree=0.0,
                        min_area_ratio=0.01,
                        scaling_ratio_range=(
                            0.09999999999999998,
                            1.9,
                        ),
                        type='YOLOv5RandomAffine',
                        use_mask_refine=True),
                ],
                prob=0.15,
                type='YOLOv5MixUp'),
            dict(keys=[
                'gt_masks',
            ], type='RemoveDataElement'),
            dict(
                bbox_params=dict(
                    format='pascal_voc',
                    label_fields=[
                        'gt_bboxes_labels',
                        'gt_ignore_flags',
                    ],
                    type='BboxParams'),
                keymap=dict(gt_bboxes='bboxes', img='image'),
                transforms=[
                    dict(p=0.01, type='Blur'),
                    dict(p=0.01, type='MedianBlur'),
                    dict(p=0.01, type='ToGray'),
                    dict(p=0.01, type='CLAHE'),
                ],
                type='mmdet.Albu'),
            dict(type='YOLOv5HSVRandomAug'),
            dict(prob=0.5, type='mmdet.RandomFlip'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'flip',
                    'flip_direction',
                ),
                type='mmdet.PackDetInputs'),
        ],
        type='YOLOv5CocoDataset'),
    num_workers=1,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_num_workers = 1
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        mask2bbox=True, type='LoadAnnotations', with_bbox=True,
        with_mask=True),
    dict(
        img_scale=(
            640,
            640,
        ),
        pad_val=114.0,
        pre_transform=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(
                mask2bbox=True,
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True),
        ],
        type='Mosaic'),
    dict(prob=0.3, type='YOLOv5CopyPaste'),
    dict(
        border=(
            -320,
            -320,
        ),
        border_val=(
            114,
            114,
            114,
        ),
        max_aspect_ratio=100.0,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        min_area_ratio=0.01,
        scaling_ratio_range=(
            0.09999999999999998,
            1.9,
        ),
        type='YOLOv5RandomAffine',
        use_mask_refine=True),
    dict(
        pre_transform=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(
                mask2bbox=True,
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True),
            dict(
                img_scale=(
                    640,
                    640,
                ),
                pad_val=114.0,
                pre_transform=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(
                        mask2bbox=True,
                        type='LoadAnnotations',
                        with_bbox=True,
                        with_mask=True),
                ],
                type='Mosaic'),
            dict(prob=0.3, type='YOLOv5CopyPaste'),
            dict(
                border=(
                    -320,
                    -320,
                ),
                border_val=(
                    114,
                    114,
                    114,
                ),
                max_aspect_ratio=100.0,
                max_rotate_degree=0.0,
                max_shear_degree=0.0,
                min_area_ratio=0.01,
                scaling_ratio_range=(
                    0.09999999999999998,
                    1.9,
                ),
                type='YOLOv5RandomAffine',
                use_mask_refine=True),
        ],
        prob=0.15,
        type='YOLOv5MixUp'),
    dict(keys=[
        'gt_masks',
    ], type='RemoveDataElement'),
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', img='image'),
        transforms=[
            dict(p=0.01, type='Blur'),
            dict(p=0.01, type='MedianBlur'),
            dict(p=0.01, type='ToGray'),
            dict(p=0.01, type='CLAHE'),
        ],
        type='mmdet.Albu'),
    dict(type='YOLOv5HSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
        ),
        type='mmdet.PackDetInputs'),
]
train_pipeline_stage2 = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        mask2bbox=True, type='LoadAnnotations', with_bbox=True,
        with_mask=True),
    dict(scale=(
        640,
        640,
    ), type='YOLOv5KeepRatioResize'),
    dict(
        allow_scale_up=True,
        pad_val=dict(img=114.0),
        scale=(
            640,
            640,
        ),
        type='LetterResize'),
    dict(
        border_val=(
            114,
            114,
            114,
        ),
        max_aspect_ratio=100,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        min_area_ratio=0.01,
        scaling_ratio_range=(
            0.09999999999999998,
            1.9,
        ),
        type='YOLOv5RandomAffine',
        use_mask_refine=True),
    dict(keys=[
        'gt_masks',
    ], type='RemoveDataElement'),
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', img='image'),
        transforms=[
            dict(p=0.01, type='Blur'),
            dict(p=0.01, type='MedianBlur'),
            dict(p=0.01, type='ToGray'),
            dict(p=0.01, type='CLAHE'),
        ],
        type='mmdet.Albu'),
    dict(type='YOLOv5HSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
        ),
        type='mmdet.PackDetInputs'),
]
tta_model = dict(
    tta_cfg=dict(max_per_img=300, nms=dict(iou_threshold=0.3, type='nms')),
    type='mmdet.DetTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(
                    transforms=[
                        dict(scale=(
                            640,
                            640,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                640,
                                640,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
                dict(
                    transforms=[
                        dict(scale=(
                            320,
                            320,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                320,
                                320,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
                dict(
                    transforms=[
                        dict(scale=(
                            960,
                            960,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                960,
                                960,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
            ],
            [
                dict(prob=1.0, type='mmdet.RandomFlip'),
                dict(prob=0.0, type='mmdet.RandomFlip'),
            ],
            [
                dict(type='mmdet.LoadAnnotations', with_bbox=True),
            ],
            [
                dict(
                    meta_keys=(
                        'img_id',
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'scale_factor',
                        'pad_param',
                        'flip',
                        'flip_direction',
                    ),
                    type='mmdet.PackDetInputs'),
            ],
        ],
        type='TestTimeAug'),
]
use_mask2refine = True
val_ann_file = 'data/pothole_coco/annotations/instances_val2017.json'
val_batch_size_per_gpu = 1
val_cfg = dict(type='ValLoop')
val_data_prefix = 'val2017/'
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/instances_val2017.json',
        batch_shapes_cfg=None,
        data_prefix=dict(img='images/'),
        data_root='data/pothole_coco/',
        metainfo=dict(classes=('cell', ), palette=[
            (
                20,
                220,
                60,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='YOLOv5CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/pothole_coco/annotations/instances_val2017.json',
    metric='bbox',
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='mmdet.CocoMetric')
val_interval_stage2 = 1
val_num_workers = 2
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
weight_decay = 0.0005
widen_factor = 1.25
work_dir = './work_dirs\\yolov8_x_mask-refine_syncbn_fast_8xb16-500e_cell'
