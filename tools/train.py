import argparse
import copy
import os
import os.path as osp
import time

import mmcv
import torch
from mmcv.runner import init_dist
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
from mmseg.apis import set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger

#配置数据集教程
#https://blog.csdn.net/weixin_44044411/article/details/118196847
def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    #parser.add_argument语句，发现当required参数为True时，在运行时必须要为该命令指定路径。第一条默认为True
    # args分为可选参数和必选参数。--指定可选参数，通过default设置默认参数。不加 --指定的是必选参数,必须手动指定的，default设置也会出错
    # ../ configs / pspnet / pspnet_r50 - d8_480x480_40k_potsdam.py
    parser.add_argument('--config', default = "../configs/pspnet/pspnet_r50-d8_480x480_40k_potsdam.py", help='train config file path')
    # , default = "/home1/xiaotao/project/mmsegmentation-master/configs/pspnet/pspnet_r50-d8_480x480_40k_potsdam.py",
    #位置参数 config，configs文件的绝对/相对路径，通常需要把工作目录设置在mmsegmentation的根目录；
    parser.add_argument('--work-dir', help='the dir to save logs and models')#可选参数，work_dir,默认会建立在根目录
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')#可选参数，resume-from， 选择用于恢复的模型，以避免从头训练。
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training') #是否在训练期间不评估检查点
    #可选参数，no-validate，action设置为store_true（命令若包含此参数，就不对模型进行训练评估，默认是要进行评估）。
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')#可选参数，gpus，非分布式训练下单机多卡的gpu数量（>0）。
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        default=5,
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')#可选参数，gpu-ids，gpu的id号 ，nargs =‘+’。命令格式为--gou-ids 0 1 2。
    parser.add_argument('--seed', type=int, default=None, help='random seed')#可选参数，seed，随机数种子数，int型
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')#可选参数，deterministic ，是否取用cudnn加速，默认没有
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')#可选参数，options，action为自定义操作
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=6)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ: # 如果环境变量中没有指定当前进程使用的GPU标号，则使用参数里指定的
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

# 单卡训练
# python tools/train.py ${config_file} [option arguments]
# 分布式训练
# ./tools/dist_train.sh ${config_file} ${gpu_num} [option arguments]

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config) #读取配置
    if args.options is not None:  # args.options是自定义操作
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    # work_dir在这个优先级中确定。CLI > 文件中的分段 > 文件名
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    # 首先启动分布式环境，因为记录器依赖于分布式信息。
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config 转储配置
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps     在其他步骤之前启动记录仪
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as environment info and seed, which will be logged
    # 启动meta dict，记录一些重要信息，如环境信息和种子，这些信息将被记录下来。
    meta = dict()
    # log env info 记录环境信息
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: '
                    f'{args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)


#######   开始训练流程    #########
    model = build_segmentor(    #创建 model
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    model.init_weights()

    logger.info(model)

    datasets = [build_dataset(cfg.data.train)]  #创建 training dataset
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset)) #创建 validation datase

    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in checkpoints as meta data
        # 将mmseg版本、配置文件内容和检查点中的类名作为元数据保存。
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE)

    # add an attribute for visualization convenience 添加一个属性以方便可视化
    model.CLASSES = datasets[0].CLASSES
    # passing checkpoint meta for saving best checkpoint 传递检查点元，保存最佳检查点
    meta.update(cfg.checkpoint_config.meta)

    train_segmentor(  #将 model, data, config 喂给训练函数:
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),#是否在训练期间不评估检查点
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
