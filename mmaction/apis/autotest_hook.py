import time
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import Hook
from mmcv.runner.dist_utils import master_only
from mmcv.runner import PaviLoggerHook
from mmaction.utils import get_root_logger

class AutoTestHook(Hook):
    def __init__(self):
        self.start_time = time.time()
        self.iter_time_list = []
        self.mem_alloc = -1
        self.mem_cached = -1
        self.pure_training_time = -1
        self.total_time = -1

    def before_run(self, runner):
        self.run_time = time.time()
        self.t = time.time()

    def after_iter(self, runner):
        cur_iter = runner.iter
        self.end_iter = time.time()
        if cur_iter >= 100 and cur_iter <= 300 and len(self.iter_time_list) < 200:
            self.iter_time_list.append(self.end_iter-self.t)
            if self.mem_alloc == -1:
                self.mem_alloc = self._get_max_memory(runner)
            if self.mem_cached == -1:
                self.mem_cached = self._get_max_memory_cached(runner)
        self.t = self.end_iter

    @master_only
    def after_run(self, runner):
        self.end_time = time.time()
        writer = None
        for hook in runner.hooks[::-1]:
            if isinstance(hook, PaviLoggerHook):
                writer = getattr(hook, 'writer', None)
        logger = get_root_logger()

        start_time = self.start_time
        run_time = self.run_time
        end_time = self.end_time
        iter_time_list = self.iter_time_list
        mem_alloc = self.mem_alloc
        mem_cached = self.mem_cached

        if writer:
            logger.info('__benchmark_total_time(h):{}'.format((end_time - start_time) / 3600))
            logger.info('__benchmark_pure_training_time(h):{}'.format((end_time - run_time) / 3600))
            logger.info('__benchmark_avg_iter_time(s):{}'.format(np.mean(iter_time_list)))
            logger.info('__benchmark_mem_alloc(mb):{}'.format(mem_alloc))
            logger.info('__benchmark_mem_cached(mb):{}'.format(mem_cached))
            writer.add_scalar('__benchmark_total_time(h)',(end_time - start_time) / 3600,1)
            writer.add_scalar('__benchmark_pure_training_time(h)',(end_time - run_time) / 3600,1)
            writer.add_scalar('__benchmark_avg_iter_time(s)',np.mean(iter_time_list),1)
            writer.add_scalar('__benchmark_mem_alloc(mb)',mem_alloc,1)
            writer.add_scalar('__benchmark_mem_cached(mb)',mem_cached,1)
            writer.add_snapshot('__benchmark_pseudo_snapshot', None, 1)

    def _get_max_memory(self, runner):
        device = getattr(runner.model, 'output_device', None)
        mem = torch.cuda.max_memory_allocated(device=device)
        mem_mb = torch.tensor([mem / (1024 * 1024)],
                              dtype=torch.int,
                              device=device)
        if runner.world_size > 1:
            dist.reduce(mem_mb, 0, op=dist.ReduceOp.MAX)
        return mem_mb.item()

    def _get_max_memory_cached(self, runner):
        device = getattr(runner.model, 'output_device', None)
        mem = torch.cuda.max_memory_cached(device=device)
        mem_mb = torch.tensor([mem / (1024 * 1024)],
                              dtype=torch.int,
                              device=device)
        if runner.world_size > 1:
            dist.reduce(mem_mb, 0, op=dist.ReduceOp.MAX)
        return mem_mb.item()
