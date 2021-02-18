import argparse
import copy
import os
import os.path as osp
import time

import mmcv
import torch
from torch import distributed as dist
from mmcv import Config
from mmcv.runner import init_dist, set_random_seed

from mmaction import __version__
from mmaction.apis import train_model
from mmaction.datasets import build_dataset
from mmaction.models import build_model
from mmaction.utils import collect_env, get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--pavi', dest='pavi', action='store_true', default=False, help='pavi use')
    parser.add_argument('--pavi-project', type=str, default="default", help='pavi project name')
    parser.add_argument('--data_reader', type=str, default="MemcachedReader", choices=['MemcachedReader', 'CephReader'], help='io backend')
    parser.add_argument('--max-step', type=int, default=None, help='training epoch')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # add pavi support
    if args.pavi:
        cfg.log_config.hooks.append(dict(type='PaviLoggerHook', init_kwargs=dict(project=args.pavi_project)))

    if args.max_step != None:
        cfg.total_epochs = args.max_step
        cfg.evaluation['interval'] = args.max_step
    
    if mmcv.__version__ <= '1.0.2' and cfg.lr_config['policy'] == 'CosineAnnealing':
        cfg.lr_config['policy'] = 'CosineAnealing'

    # add ceph support
    if args.data_reader=='CephReader':
        cfg.data_root = cfg.ceph_data_root
        cfg.data_root_val = cfg.ceph_data_root_val
        cfg.ann_file_train = cfg.ceph_ann_file_train
        cfg.ann_file_val = cfg.ceph_ann_file_val
        cfg.ann_file_test = cfg.ceph_ann_file_test

        ceph_dict = {
            'io_backend':'petrel'
        }
        cfg.train_pipeline[0].update(ceph_dict)
        cfg.val_pipeline[0].update(ceph_dict)
        cfg.test_pipeline[0].update(ceph_dict)

        cfg.data.train.ann_file = cfg.ann_file_train
        cfg.data.train.data_prefix = cfg.data_root
        cfg.data.train.pipeline = cfg.train_pipeline

        cfg.data.val.ann_file = cfg.ann_file_val
        cfg.data.val.data_prefix = cfg.data_root_val
        cfg.data.val.pipeline = cfg.val_pipeline

        cfg.data.test.ann_file = cfg.ann_file_test
        cfg.data.test.data_prefix = cfg.data_root_val
        cfg.data.test.pipeline = cfg.test_pipeline


    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority:
    # CLI > config file > default (base filename)
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        if args.launcher == "slurm":
            init_dist(args.launcher, **cfg.dist_params)
        elif args.launcher == "mpi":
            rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
            num_gpus = torch.cuda.device_count()
            torch.cuda.set_device(rank % num_gpus)
            #dist.init_process_group(backend="nccl", **cfg.dist_params)
            #dist.init_process_group(**cfg.dist_params)
            dist.init_process_group(backend="nccl")
        else:
            pass

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config: {cfg.text}')

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}, deterministic: {}'.format(
            args.seed, args.deterministic))
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed

    model = build_model(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmaction version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmaction_version=__version__, config=cfg.text)

    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=args.validate,
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
