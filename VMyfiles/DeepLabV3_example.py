# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)  # 分割通常使用SyncBN
model = dict(
    type='EncoderDecoder',  # 分割器名称
    pretrained='open-mmlab://resnet50_v1c',  # 加载ImageNet预训练的backbone
    backbone=dict(
        type='ResNetV1c',  # backbone的类型
        depth=50,  # backbone的深度，一般用50，101
        num_stages=4,  # backbone的阶段数
        out_indices=(0, 1, 2, 3),  # 每个阶段生成的输出特征图的索引
        dilations=(1, 1, 2, 4),  # 每层的膨胀率
        strides=(1, 2, 1, 1),  # 每层的步幅
        norm_cfg=norm_cfg,  # 规范层的配置
        norm_eval=False,  # 是否冻结BN中的统计信息
        style='pytorch',  # 主干的样式，“ pytorch”表示第2步的步幅为3x3卷积，“ caffe”表示第2步的步幅为1x1卷积
        contract_dilation=True),  # 当扩张> 1时，是否收缩第一层扩张。
    decode_head=dict(
        type='ASPPHead',  # 解码器的类型，请参阅mmseg / models / decode_heads了解可用选项。
        in_channels=2048,  # 解码器头输入通道数
        in_index=3,  # 选择特征图的索引
        channels=512,  # 解码器头的中间通道数
        dilations=(1, 12, 24, 36),
        dropout_ratio=0.1,  # 在最终分类层之前的dropout率
        num_classes=19,  # 分割类别的数量，通常对于城市景观为19，对于VOC为21，对于ADE20k为150。
        norm_cfg=norm_cfg,
        align_corners=False,  # 用于在解码时调整大小
        loss_decode=dict(  # 解码器损失函数的配置
            type='CrossEntropyLoss',  # 用于分割的损失函数的类型
            use_sigmoid=False,  # 是否使用sigmode激活函数
            loss_weight=1.0)),  # 解码器的损失权重
    auxiliary_head=dict(
        type='FCNHead',  # auxiliary_head（辅助头）的类型，请参阅mmseg / models / decode_heads了解可用选项。
        in_channels=1024,  # 辅助头的输入通道数
        in_index=2,  # 选择特征图索引
        channels=256,  # 解码器的中间通道
        num_convs=1,  # FCNHead的卷积数，在auxiliary_head通常是1
        concat_input=False,  # 是否将convs的输出与分类器之前的输入进行拼接
        dropout_ratio=0.1,  # 最终分类器之前的dropout率
        num_classes=19,  #
        norm_cfg=norm_cfg,  # norm层的配置
        align_corners=False,  # 用于在解码时调整大小
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4)),  # Loss weight of auxiliary head, which is usually 0.4 of decode head
    # model training and testing settings
    train_cfg=dict(),  # train_cfg目前仅是一个占位符
    test_cfg=dict(mode='whole'))  # 测试模块，‘whole’：整个图像全卷积测试，‘sliding’：裁剪图像

# dataset settings
dataset_type = 'CityscapesDataset'  # 数据集类型，将用于定义数据集
data_root = 'data/cityscapes/'  # 数据的根路径
img_norm_cfg = dict(  # 图像归一化配置以对输入图像进行归一化
    mean=[123.675, 116.28, 103.53],  # 用于预训练预训练骨干模型的平均值
    std=[58.395, 57.12, 57.375],  # 用于预训练预训练骨干模型的标准方差
    to_rgb=True)  # 用于预训练预训练骨干模型的图像通道顺序
crop_size = (512, 1024)  # 训练时的图像剪裁大小
train_pipeline = [  # 训练通道
    dict(type='LoadImageFromFile'),  # 从文件加载图像的第一个通道
    dict(type='LoadAnnotations'),  # 为当前图像加载注释的第二个通道
    dict(type='Resize',  # 增强通道以调整图像的大小及其注释
         img_scale=(2048, 1024),  # 最大图像比例
         ratio_range=(0.5, 2.0)),  # 扩大比率的范围
    dict(type='RandomCrop',  # 增强通道从当前图像中随机裁剪一patch
         crop_size=crop_size,  # patch的大小
         cat_max_ratio=0.75),  # 单个类别可以占用的最大面积比
    dict(type='RandomFlip',  # 翻转图像及其注释的增强通道
         prob=0.5),  # 翻转的比例或概率
    dict(type='PhotoMetricDistortion'),  # 增强通道，通过多种光度学方法使当前图像失真
    dict(type='Normalize',  # 标准化输入图像的增强通道
         **img_norm_cfg),
    dict(type='Pad',  # 将图像填充到指定大小的增强通道
         size=crop_size,  # 填充的输出大小
         pad_val=0,  # 图片的填充值
         seg_pad_val=255),  # 'gt_semantic_seg'的填充值
    dict(type='DefaultFormatBundle'),  # 默认格式捆绑包，用于收集管道中的数据
    dict(type='Collect',  # 决定应将数据中的哪些键传递给分割器的管道
         keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),  # 第一个从文件路径加载图像的管道
    dict(
        type='MultiScaleFlipAug',  # 封装测试时间扩展的封装
        img_scale=(2048, 1024),  # 确定用于调整管道大小的最大测试规模
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],#更改为多规模测试
        flip=False,  # 测试期间是否翻转图像
        transforms=[
            dict(type='Resize',  # 使用调整大小的增强Use resize augmentation
                 keep_ratio=True),  # 是否保持高宽比，此处设置的img_scale将被以上设置的img_scale取代
            dict(type='RandomFlip'),  # 以为RandomFlip是在管道中添加的，当flip = False时不使用
            dict(type='Normalize',  # 规范化配置，值来自img_norm_cfg
                 **img_norm_cfg),
            dict(type='ImageToTensor',  # 将图像转换为张量
                 keys=['img']),
            dict(type='Collect',  # 收集必要密钥以进行测试的Collect管道
                 keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,  # 单个GPU的批处理大小
    workers_per_gpu=2,  # 为每个GPU预取数据
    train=dict(  # 训练数据集配置
        type=dataset_type,  # 数据集类型，有关详细信息，请参考mmseg / datasets /
        data_root=data_root,  # 数据集的位置
        img_dir='leftImg8bit/train',  # 数据集的图像目录
        ann_dir='gtFine/train',  # 数据集的注释目录
        pipeline=train_pipeline),  # 管道，这是由之前创建的train_pipeline传递的
    val=dict(
        type=dataset_type,  # 验证数据集配置
        data_root=data_root,
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline),  # 管道由之前创建的test_pipeline传递
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline))

# yapf:disable
log_config = dict(  # 配置注册记录器钩子
    interval=50,  # 间隔打印日志
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')#还支持Tensorboard记录器
    ])
# yapf:enable
dist_params = dict(backend='nccl')  # 参数设置分布式训练，也可以设置端口
log_level = 'INFO'  # 日志级别
load_from = None  # 从给定路径将模型加载为预训练模型。 这不会恢复训练
resume_from = None  # 从给定路径恢复检查点，保存检查点后，训练将从迭代中恢复
workflow = [('train',
             1)]  # Workflow for runner [（'train'，1）]表示只有一个工作流程，名为'train'的工作流程只执行一次。
# 工作流程根据`runner.max_iters`进行了40000次迭代训练模型。
cudnn_benchmark = True  # 是否使用cudnn_benchmark加快速度，只适用于固定输入大小

# optimizer
optimizer = dict(  # 用于构建优化器的Config，支持PyTorch中的所有优化器，其参数也与PyTorch中的参数相同
    type='SGD',  # 优化程序的类型，
    # 请参阅https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/optimizer/default_constructor.py#L13了解更多信息
    lr=0.01,  # 优化器的学习率，请参阅PyTorch文档中参数的详细用法
    momentum=0.9,  # 动量
    weight_decay=0.0005)  # SGD的重量衰减
optimizer_config = dict()  # 用于构建优化程序挂钩的配置，
# 有关实现的详细信息，请参阅https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/optimizer.py#L8。
# learning policy
lr_config = dict(policy='poly',
                 # scheduler的策略还支持Step，CosineAnnealing，Cyclic等。
                 # 请参阅https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#所支持的LrUpdater的详细信息。 L9。
                 power=0.9,  # 多项式衰减的幂
                 min_lr=1e-4,  # 稳定训练的最低学习率
                 by_epoch=False)  # 是否按epoch计数
# runtime settings
runner = dict(type='IterBasedRunner',  # 要使用的运行程序类型（即IterBasedRunner或EpochBasedRunner）
              max_iters=40000)  # 迭代总数。 对于EpochBasedRunner使用`max_epochs`
checkpoint_config = dict(
    # 进行配置以设置检查点挂钩，有关实现请参阅https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py
    by_epoch=False,  # 是否按epoch计数
    interval=4000)  # 保存间隔
evaluation = dict(  # 用于构建评估挂钩的配置。 有关详细信息，请参考mmseg / core / evaulation / eval_hook.py
    interval=4000,  # 评估间隔
    metric='mIoU')  # 评估指标