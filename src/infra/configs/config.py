from mmdet.utils import (replace_cfg_vals, setup_multi_processes, update_data_root)
from mmdet.utils import (collect_env, get_root_logger, get_device, compat_cfg)
from mmdet.apis import init_random_seed, set_random_seed
from mmdet.datasets import replace_ImageToTensor
from dataclasses import dataclass, field
from mmcv import Config
import os.path as osp
import time
import json
import mmcv
import os


@dataclass
class Configuration:
    base_file: str
    config_file: str = field(default=None)
    cfg: Config = field(default=None)

    def __post_init__(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        with open(f"{base_path}/run/config_base.json") as f:
            config_base = json.load(f)

        with open(f"{base_path}/run/{self.base_file}.json") as f:
            file = json.load(f)

        self.config_file = f"""../../infra/configs/base/{file["model"]["name"]}/{file["fine_tune"]["name"]}_{file["dataset"]["name"]}.py"""
        cfg = Config.fromfile(self.config_file)

        cfg.load_from = file["fine_tune"]["load_from"]
        cfg.work_dir = f"""../../../{config_base["base_work_dir"]}/{file["model"]["name"]}/{file["model"]["version"]}/{file["dataset"]["name"]}"""

        cfg = replace_cfg_vals(cfg)

        # update data root according to MMDET_DATASETS
        update_data_root(cfg)

        # set multi-process settings
        setup_multi_processes(cfg)

        cfg.optimizer.lr = file["model"]["optimizer"]["lr"]
        cfg.optimizer.momentum = file["model"]["optimizer"]["momentum"]

        cfg.checkpoint_config.interval = file["model"]["optimizer"]["interval"]

        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

        # Set random seed for reproducible results.
        seed = init_random_seed(0, device=cfg.device)
        set_random_seed(seed)
        cfg.seed = seed
        cfg.runner.max_epochs = file["model"]["runner"]["max_epochs"]

        # dump config
        cfg.dump(osp.join(cfg.work_dir, osp.basename(self.config_file)))

        self.cfg = cfg

    def load_config_for_train(self) -> dict:
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(self.cfg.work_dir, f'{timestamp}.log')
        logger = get_root_logger(log_file=log_file, log_level=self.cfg.log_level)

        # init the meta dict to record some important information such as
        # environment info and seed, which will be logged
        meta = dict()

        # log env info
        env_info_dict = collect_env()
        env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)

        meta['env_info'] = env_info
        meta['config'] = self.cfg.pretty_text

        # log some basic info
        logger.info(f'Distributed training: {False}')
        logger.info(f'Config:\n{self.cfg.pretty_text}')

        # set random seeds
        meta['seed'] = self.cfg.seed

        return {
            "cfg": self.cfg,
            "timestamp": timestamp,
            "meta": meta
        }

    def load_config_for_test(self) -> Config:
        self.cfg = replace_cfg_vals(self.cfg)

        # update data root according to MMDET_DATASETS
        update_data_root(self.cfg)
        self.cfg = compat_cfg(self.cfg)

        if 'pretrained' in self.cfg.model:
            self.cfg.model.pretrained = None
        elif 'init_cfg' in self.cfg.model.backbone:
            self.cfg.model.backbone.init_cfg = None

        if self.cfg.model.get('neck'):
            if isinstance(self.cfg.model.neck, list):
                for neck_cfg in self.cfg.model.neck:
                    if neck_cfg.get('rfp_backbone'):
                        if neck_cfg.rfp_backbone.get('pretrained'):
                            neck_cfg.rfp_backbone.pretrained = None
            elif self.cfg.model.neck.get('rfp_backbone'):
                if self.cfg.model.neck.rfp_backbone.get('pretrained'):
                    self.cfg.model.neck.rfp_backbone.pretrained = None

        self.cfg.device = get_device()

        if isinstance(self.cfg.data.test, dict):
            self.cfg.data.test.test_mode = True
            if self.cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
                # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                self.cfg.data.test.pipeline = replace_ImageToTensor(self.cfg.data.test.pipeline)
        elif isinstance(self.cfg.data.test, list):
            for ds_cfg in self.cfg.data.test:
                ds_cfg.test_mode = True
            if self.cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
                for ds_cfg in self.cfg.data.test:
                    ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

        return self.cfg

