from mmdet.apis import init_random_seed, set_random_seed
from mmdet.utils import (replace_cfg_vals, setup_multi_processes, update_data_root)
import os.path as osp
from dataclasses import dataclass
from mmcv import Config
import json
import mmcv


@dataclass
class Configuration:

    @staticmethod
    def parse_file_to_config(file: str) -> Config:
        with open("configs/config_base.json") as f:
            config_base = json.load(f)

        with open(f"configs/{file}.json") as f:
            file = json.load(f)

        config = f"""base/configs/{file["model"]["name"]}/{file["fine_tune"]["name"]}_{config_base["project_name"]}.py"""
        cfg = Config.fromfile(config)

        cfg.load_from = file["fine_tune"]["load_from"]
        cfg.work_dir = f"""../{config_base["base_work_dir"]}/{file["model"]["name"]}/{file["dataset"]["name"]}"""

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
        cfg.dump(osp.join(cfg.work_dir, osp.basename(config)))

        return cfg
