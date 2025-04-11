#########################
# Basic Global Settings #
#########################

point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
voxel_size = [0.075, 0.075, 0.2]   # Still referenced by the bbox coder for 3D boxes
out_size_factor = 8               # Affects downsampling factor; still used in some heads
evaluation = dict(interval=1)
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'

# Use camera only
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False
)

img_scale = (800, 448)
num_views = 6
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

#########################
# Data Pipeline - TRAIN #
#########################

# 1) Remove LiDAR loading and sweeps.
# 2) Keep only camera loading & augmentation steps.
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles'),  # Loads images

    # Insert our GenerateCenter2DHeatmap pipeline step
    dict(type='GenerateCenter2DHeatmap',
         ann_file_2d='data/nuscenes/nuscenes_2dcenters.json',
         downscale=4,
         sigma=2,
         image_size=(900,1600)),  

    # Optionally apply 3D transforms if you have custom transforms for cameras.
    # (Below are commented out for clarity, you can enable if your code supports them):
    # dict(
    #     type='GlobalRotScaleTrans',
    #     rot_range=[-0.3925 * 2, 0.3925 * 2],
    #     scale_ratio_range=[0.9, 1.1],
    #     translation_std=[0.5, 0.5, 0.5]),
    # dict(
    #     type='RandomFlip3D',
    #     sync_2d=True,
    #     flip_ratio_bev_horizontal=0.5,
    #     flip_ratio_bev_vertical=0.5),

    # Remove point-based filtering since we do not use LiDAR.
    # dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='ObjectNameFilter', classes=class_names),
    # dict(type='PointShuffle'),

    # Load your 3D annotations for camera-based training:
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),

    # Camera-specific resizing, normalization, padding:
    # dict(type='MyResize', img_scale=img_scale, keep_ratio=True),
    dict(type='MyNormalize', **img_norm_cfg),
    dict(type='MyPad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=['sample_idx', 'img_filename', 'lidar2img']
    )
]

########################
# Data Pipeline - TEST #
########################
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=img_scale,
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            # For testing, you typically disable large random transforms:
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]
            ),
            dict(type='RandomFlip3D'),
            dict(type='MyResize', img_scale=img_scale, keep_ratio=True),
            dict(type='MyNormalize', **img_norm_cfg),
            dict(type='MyPad', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False
            ),
            dict(type='Collect3D', keys=['img'])
        ]
    )
]

###########
# Datasets #
###########
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            num_views=num_views,
            ann_file=data_root + '/nuscenes_infos_train.pkl',
            load_interval=1,
            pipeline=train_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            box_type_3d='LiDAR'  # nuScenes boxes commonly labeled in LiDAR coords
        )
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        num_views=num_views,
        ann_file=data_root + '/nuscenes_infos_val.pkl',
        load_interval=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        num_views=num_views,
        ann_file=data_root + '/nuscenes_infos_val.pkl',
        load_interval=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'
    )
)

#########
# Model #
#########
model = dict(
    type='TransFusionDetector',
    
    # Freeze this if you want or set to False
    # This param is originally for freezing camera backbone in fusion.
    freeze_img=False,

    # -------------------------------
    # Camera backbone & neck
    # -------------------------------
    img_backbone=dict(
       type='DLASeg',  # Or a ResNet, etc.
       num_layers=34,
       heads={},
       head_convs=-1,
    ),
    img_neck=dict(
        type='FPN',
        in_channels=[64,128,256,512],
        out_channels=256,
        num_outs=4
    ),

    # -----------------------------------------------------
    # Remove (or set to None) all LiDAR-specific modules
    # -----------------------------------------------------
    pts_voxel_layer=None,
    pts_voxel_encoder=None,
    pts_middle_encoder=None,
    pts_backbone=None,
    pts_neck=None,

    # -----------------------------------------
    # The 3D head that originally fuses LiDAR+img
    # We adapt it to camera-only.
    # -----------------------------------------
    pts_bbox_head=dict(
        type='TransFusionHead',

        # Since we have no LiDAR feature maps, set fuse_img=False
        fuse_img=False,

        # We still want to process multiple camera views
        num_views=num_views,

        # Channels from your chosen img_backbone (e.g., DLA, FPN)
        # Adjust in_channels_img according to your backbone output.
        in_channels_img=64,  

        # Decrease num_proposals or keep the same.
        # We must generate proposals from camera features only, so set:
        initialize_by_heatmap=True,  # if you want the head to do top-K from a heatmap
        num_proposals=200,

        # The rest are normal settings for a 3D head:
        auxiliary=False,  # No LiDAR-based auxiliary losses
        in_channels=256,  # Not used effectively if fuse_img=False, but keep for code consistency
        hidden_channel=128,
        num_classes=len(class_names),
        num_decoder_layers=1,
        num_heads=8,
        learnable_query_pos=False,
        nms_kernel_size=3,
        ffn_channel=256,
        dropout=0.1,
        bn_momentum=0.1,
        activation='relu',
        common_heads=dict(
            center=(2, 2),
            height=(1, 2),
            dim=(3, 2),
            rot=(2, 2),
            vel=(2, 2)
        ),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=point_cloud_range[:2],        # [-54.0, -54.0]
            voxel_size=voxel_size[:2],            # [0.075, 0.075]
            out_size_factor=out_size_factor,      # 8
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.0,
            code_size=10
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2,
            alpha=0.25,
            reduction='mean',
            loss_weight=1.0
        ),
        loss_bbox=dict(
            type='L1Loss',
            reduction='mean',
            loss_weight=0.25
        ),
        loss_heatmap=dict(
            type='GaussianFocalLoss',
            reduction='mean',
            loss_weight=1.0
        ),
    ),

    # -------------------------------
    # 2D Center Heatmap Head
    # -------------------------------

    center2d_head=dict(
        type='Center2DHead',
        in_channels=256,  # match your final FPN scale
        num_classes=1,
        loss_center=dict(
            type='GaussianFocalLoss',
            loss_weight=1.0
        )
    ),

    # -------------------------------
    # Depth Head
    # -------------------------------

    # depth_head=dict(
    #     type='DepthHead',
    #    in_channels=256,       # match final FPN output channels
    #    mid_channels=128,
    #    depth_channels=1,
    #    loss_depth=dict(
    #        type='MSELoss',
    #        loss_weight=1.0
    #    )
    # ),

    # -----------------------------
    # Training settings
    # (No LiDAR-based assigned needed, but we
    # keep the structure for 3D boxes in 'lidar' coords.)
    # -----------------------------
    train_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            assigner=dict(
                type='HungarianAssigner3D',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(type='FocalLossCost', gamma=2, alpha=0.25, weight=0.15),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                iou_cost=dict(type='IoU3DCost', weight=0.25)
            ),
            pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            grid_size=[1440, 1440, 40],
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            point_cloud_range=point_cloud_range
        )
    ),
    test_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            grid_size=[1440, 1440, 40],
            out_size_factor=out_size_factor,
            pc_range=point_cloud_range[0:2],
            voxel_size=voxel_size[:2],
            nms_type=None
        )
    )
)

##############
# Optimizers #
##############
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.01)
optimizer_config = dict(
    type='GradientCumulativeOptimizerHook',    
    cumulative_iters=4,                       # e.g. accumulate 4 mini-batches
    grad_clip=dict(max_norm=0.1, norm_type=2)
)

# Learning rate schedule
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 0.0001),
    cyclic_times=1,
    step_ratio_up=0.4
)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4
)
total_epochs = 20

checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = 'checkpoints/ctdet_coco_dla_2x.pth'
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(1)
freeze_lidar_components = True
find_unused_parameters = True