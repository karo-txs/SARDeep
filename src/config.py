from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.utils import (collect_env, get_device, get_root_logger,
                         replace_cfg_vals, rfnext_init_model,
                         setup_multi_processes, update_data_root)
import os.path as osp
from dataclasses import dataclass
from mmcv import Config
import json
import mmcv


@dataclass
class Configuration:
    @staticmethod
    def parse_file_to_config(file: json):
        config = f"""configs/{file["project_name"]}/{file["fine_tune"]["name"]}_{file["project_name"]}.py"""
        cfg = Config.fromfile(config)

        cfg.load_from = file["fine_tune"]["load_from"]
        cfg.work_dir = file["work_dir"]

        cfg = replace_cfg_vals(cfg)

        # update data root according to MMDET_DATASETS
        update_data_root(cfg)

        # set multi-process settings
        setup_multi_processes(cfg)

        cfg.optimizer.lr = file["optimizer"]["learning_rate"]
        cfg.optimizer.momentum = file["optimizer"]["momentum"]

        cfg.checkpoint_config.interval = file["optimizer"]["interval"]

        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

        # Set random seed for reproducible results.
        seed = init_random_seed(0, device=cfg.device)
        set_random_seed(seed)
        cfg.seed = seed
        cfg.runner.max_epochs = file["runner"]["max_epochs"]

        # dump config
        cfg.dump(osp.join(cfg.work_dir, osp.basename(config)))
