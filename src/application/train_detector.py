from mmdet.utils import (collect_env, get_root_logger)
from src.infra.configs.config import Configuration
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmcv.utils import get_git_hash
from dataclasses import dataclass
from mmdet import __version__
import os.path as osp
import argparse
import time


@dataclass
class Train:
    config_file: str

    def __post_init__(self):
        self.cfg = Configuration(self.config_file).cfg
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file',  help='configuration path name')
    args = vars(parser.parse_args())

    train = Train("config_ssd_sard_v1")
    train.train_model()
