from src.config import Configuration
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.utils import (collect_env, get_device, get_root_logger,
                         replace_cfg_vals, rfnext_init_model,
                         setup_multi_processes, update_data_root)
from mmcv.utils import get_git_hash
from mmdet import __version__
import os.path as osp
import time


class Train:
    config_file: str

    def __post_init__(self):
        self.cfg = Configuration.parse_file_to_config(self.config_file)
        self.timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(self.cfg.work_dir, f'{self.timestamp}.log')
        logger = get_root_logger(log_file=log_file, log_level=self.cfg.log_level)

        # init the meta dict to record some important information such as
        # environment info and seed, which will be logged
        self.meta = dict()

        # log env info
        env_info_dict = collect_env()
        env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)

        self.meta['env_info'] = env_info
        self.meta['config'] = self.cfg.pretty_text

        # log some basic info
        logger.info(f'Distributed training: {False}')
        logger.info(f'Config:\n{self.cfg.pretty_text}')

        # set random seeds
        self.meta['seed'] = self.cfg.seed
        # meta['exp_name'] = osp.basename(args.config)

    def train_model(self):
        model = build_detector(self.cfg.model)
        model.init_weights()

        # Build dataset
        datasets = [build_dataset(self.cfg.data.train)]

        model.CLASSES = datasets[0].CLASSES
        self.cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)

        train_detector(model, datasets, self.cfg, distributed=False, validate=True, timestamp=self.timestamp,
                       meta=self.meta)
