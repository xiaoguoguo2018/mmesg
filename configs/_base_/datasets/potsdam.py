# dataset settings
dataset_type = 'PotsdamDataset'
data_root = '/home1/xiaotao/dataset/potsdam_processing/'
img_norm_cfg = dict(        #数据集的方差和均值
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

img_scale = (256, 256) #原图像尺寸
# crop_size = (480, 480) #数据增强时裁剪的大小. img_dir

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    # dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    # dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            # dict(type='Resize', keep_ratio=True),
            # dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=4,  #batch size
    workers_per_gpu=4,  #workers_per_gpu：dataloader的线程数目，一般设2，4，8，根据CPU核数确定
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train',  #工作目录下存图片的目录
        ann_dir='train_labels', #工作目录下存标签的目录/home1/xiaotao/dataset/potsdam_processing/splits/train.txt
        split='splits/train.txt', #之前操作做txt文件的目录
        pipeline=train_pipeline),

    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test',
        ann_dir='test_labels',
        split='splits/val.txt',
        pipeline=test_pipeline),

    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test',
        ann_dir='test_labels',
        split='splits/val.txt',
        pipeline=test_pipeline))
