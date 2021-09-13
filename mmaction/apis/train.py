import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner, OptimizerHook,
                         build_optimizer)

from ..core import DistEvalHook, EvalHook, Fp16OptimizerHook
from ..datasets import build_dataloader, build_dataset
from ..utils import get_root_logger
from .autotest_hook import AutoTestHook

try:
    import io
    import os
    from mmcv.runner import CheckpointLoader, get_dist_info
    from mmcv.fileio import FileClient
    @CheckpointLoader.register_scheme(prefixes='s3://', force=True)
    def load_from_ceph(filename, map_location=None, backend='petrel'):
        """load checkpoint through the file path prefixed with s3. In distributed
        setting, this function only download checkpoint at local rank 0.
        Args:
            filename (str): checkpoint file path with s3 prefix
            map_location (str, optional): Same as :func:`torch.load`.
            backend (str): The storage backend type. Options are "disk", "ceph",
                "memcached" and "lmdb". Default: 'ceph'
        Returns:
            dict or OrderedDict: The loaded checkpoint.
        """
        rank, world_size = get_dist_info()
        rank = int(os.environ.get('LOCAL_RANK', rank))
        allowed_backends = ['ceph', 'petrel']
        if backend not in allowed_backends:
            raise ValueError(f'Load from Backend {backend} is not supported.')
        if rank == 0:
            fileclient = FileClient(backend=backend)
            buffer = io.BytesIO(fileclient.Get(filename))
            checkpoint = torch.load(buffer, map_location=map_location)
        if world_size > 1:
            torch.distributed.barrier()
            if rank > 0:
                fileclient = FileClient(backend=backend)
                buffer = io.BytesIO(fileclient.Get(filename))
                checkpoint = torch.load(buffer, map_location=map_location)
        return checkpoint
except:
    pass


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None):
    """Train model entry function.

    Args:
        model (nn.Module): The model to be trained.
        dataset (:obj:`Dataset`): Train dataset.
        cfg (dict): The config dict for training.
        distributed (bool): Whether to use distributed training.
            Default: False.
        validate (bool): Whether to do evaluation. Default: False.
        timestamp (str | None): Local time for runner. Default: None.
        meta (dict | None): Meta dict to record some important information.
            Default: None
    """
    logger = get_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', {}),
        workers_per_gpu=cfg.data.get('workers_per_gpu', {}),
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('train_dataloader', {}))

    data_loaders = [
        build_dataloader(ds, **dataloader_setting) for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = EpochBasedRunner(
        model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    if distributed:
        runner.register_hook(DistSamplerSeedHook())

    # register autotest hook
    runner.register_hook(AutoTestHook())

    if validate:
        eval_cfg = cfg.get('evaluation', {})
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        dataloader_setting = dict(
            videos_per_gpu=cfg.data.get('videos_per_gpu', {}),
            workers_per_gpu=cfg.data.get('workers_per_gpu', {}),
            # cfg.gpus will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            shuffle=False)
        dataloader_setting = dict(dataloader_setting,
                                  **cfg.data.get('val_dataloader', {}))
        val_dataloader = build_dataloader(val_dataset, **dataloader_setting)
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        if cfg.load_from.startswith("s3://"):
            from petrel_client.client import Client
            from mmcv.runner import load_state_dict
            import io
            file_ceph = io.BytesIO(Client().Get(cfg.load_from))
            checkpoint = torch.load(file_ceph,map_location="cpu")
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            # load state_dict
                load_state_dict(runner.model, state_dict, strict=False, logger=logger)
        else:
           runner.load_checkpoint(cfg.load_from)

    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
