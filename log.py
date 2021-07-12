import logging
import os
import re
import shutil
import sys
from typing import Tuple

from tensorboardX import SummaryWriter

# from seatable import STLogger
from utils import time_str


def create_logger(logname, filename, level=logging.INFO, stream=True):
    l = logging.getLogger(logname)
    formatter = logging.Formatter(
        fmt='[%(asctime)s][%(filename)10s][line:%(lineno)4d][%(levelname)4s] %(message)s',
        datefmt='%m-%d %H:%M:%S'
    )
    fh = logging.FileHandler(filename)
    fh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    if stream:
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setFormatter(formatter)
        l.addHandler(sh)
    return l


class DistLogger(object):
    def __init__(self, lg, verbose):
        self._lg, self._verbose = lg, verbose
    
    @staticmethod
    def do_nothing(*args, **kwargs):
        pass
    
    def __getattr__(self, attr: str):
        return getattr(self._lg, attr) if self._verbose else DistLogger.do_nothing
    
    def __del__(self):
        if self._lg is not None and hasattr(self._lg, 'close'):
            self._lg.close()


def create_loggers(prj_root, sh_root, exp_root, dist) -> Tuple[logging.Logger, SummaryWriter]:
    # create the exp dir
    if dist.is_master():
        os.makedirs(exp_root)
        
        # backup scripts
        back_dir = os.path.join(exp_root, 'back_up')
        shutil.copytree(
            src=prj_root, dst=back_dir,
            ignore=shutil.ignore_patterns('.*', '*ckpt*', '*exp*', '__pycache__'),
            ignore_dangling_symlinks=True
        )
        shutil.copytree(
            src=sh_root, dst=back_dir + sh_root.replace(prj_root, ''),
            ignore=lambda _, names: {n for n in names if not re.match(r'^(.*)\.(yaml|sh)$', n)},
            ignore_dangling_symlinks=True
        )
        print(f'{time_str()}[rk00] => All the scripts are backed up to \'{back_dir}\'.\n')
    dist.barrier()
    
    # create loggers
    exp_name = os.path.split(sh_root)[-1]
    logger = create_logger('G', os.path.join(exp_root, 'log.txt')) if dist.is_master() else None
    # seatable_logger = STLogger(exp_root, exp_name) if dist.is_master else None
    
    if dist.is_master():
        os.mkdir(os.path.join(exp_root, 'events'))
    dist.barrier()
    tensorboard_logger = SummaryWriter(os.path.join(exp_root, 'events', f'rk{dist.rank}'))
    
    return (
        DistLogger(logger, verbose=dist.is_master()),
        # DistLogger(seatable_logger, verbose=dist.is_master()),
        DistLogger(tensorboard_logger, verbose=True),
    )
