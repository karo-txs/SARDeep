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
    base_file: dict
    cfg: Config = field(default=None)

    def __post_init__(self):
        config_file = f"""../src/infra/configs/{self.base_file["name"]}/{self.base_file["fine_tune"]["name"]}.py"""
        self.cfg = Config.fromfile(config_file)

        self.cfg.load_from = self.base_file["fine_tune"]["load_from"]
        self.cfg.work_dir = f"""../../results/{self.base_file["name"]}/{self.base_file["version"]}/{self.base_file["datasets"]["train"]}/{self.base_file["datasets"]["test"]}"""

        self.config_dataset()

        self.cfg = replace_cfg_vals(self.cfg)

        # update data root according to MMDET_DATASETS
        update_data_root(self.cfg)

        # set multi-process settings
        setup_multi_processes(self.cfg)

        self.cfg.optimizer.lr = self.base_file["optimizer"]["lr"]
        self.cfg.optimizer.momentum = self.base_file["optimizer"]["momentum"]

        self.cfg.checkpoint_config.interval = self.base_file["optimizer"]["interval"]

        mmcv.mkdir_or_exist(osp.abspath(self.cfg.work_dir))

        # Set random seed for reproducible results.
        seed = init_random_seed(0, device=self.cfg.device)
        set_random_seed(seed)
        self.cfg.seed = seed
        self.cfg.runner.max_epochs = self.base_file["runner"]["max_epochs"]

        # dump config
        self.cfg.dump(osp.join(self.cfg.work_dir, osp.basename(config_file)))

    def config_dataset(self):
        data_root = 'mmdetection/data'
        ann_file = "VOC2012/ImageSets/Main"
        img_prefix = "VOC2012"

        train_data_path = f"""{data_root}/{self.base_file["datasets"]["train"]}"""
        self.cfg.data_root = train_data_path

        self.cfg.data.train.ann_file = f"""{train_data_path}/{ann_file}/train.txt"""
        self.cfg.data.train.img_prefix = f"""{train_data_path}/{img_prefix}"""

        test_data_path = f"""{data_root}/{self.base_file["datasets"]["test"]}"""
        self.cfg.data.test.ann_file = f"""{test_data_path}/{ann_file}/test.txt"""
        self.cfg.data.test.img_prefix = f"""{test_data_path}/{img_prefix}"""

        val_data_path = f"""{data_root}/{self.base_file["datasets"]["val"]}"""
        self.cfg.data.val.ann_file = f"""{val_data_path}/{ann_file}/val.txt"""
        self.cfg.data.val.img_prefix = f"""{val_data_path}/{img_prefix}"""

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

