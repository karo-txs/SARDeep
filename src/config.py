from mmdet.utils import (replace_cfg_vals, setup_multi_processes, update_data_root)
from mmdet.apis import init_random_seed, set_random_seed
from dataclasses import dataclass, field
from mmcv import Config
import os.path as osp
import json
import mmcv


@dataclass
class Configuration:
    base_file: str
    config_file: str = field(default=None)
    cfg: Config = field(default=None)

    def __post_init__(self):
        with open("runner_configs/config_base.json") as f:
            config_base = json.load(f)

        with open(f"runner_configs/{self.base_file}.json") as f:
            file = json.load(f)

        self.config_file = f"""base/configs/{file["model"]["name"]}/{file["fine_tune"]["name"]}_{file["dataset"]["name"]}.py"""
        cfg = Config.fromfile(self.config_file)

        cfg.load_from = file["fine_tune"]["load_from"]
        cfg.work_dir = f"""../{config_base["base_work_dir"]}/{file["model"]["name"]}/{file["model"]["version"]}/{file["dataset"]["name"]}"""

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
