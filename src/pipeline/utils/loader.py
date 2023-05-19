from mmdet.datasets import build_dataset, build_dataloader
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass, field
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from typing import Tuple
from mmcv import Config
import torch.nn as nn


@dataclass
class Loader:
    cfg: Config = field(default=None)

    def load_dataset(self, split: str = "test_dataloader") -> Tuple[Dataset, DataLoader]:
        dataloader_default_args = dict(samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)

        loader_cfg = {
            **dataloader_default_args,
            **self.cfg.data.get(split, {})
        }

        dataset = build_dataset(self.cfg.data.test)
        data_loader = build_dataloader(dataset, **loader_cfg)
        return dataset, data_loader

    def load_model(self, load: str = "latest") -> nn.Module:
        self.cfg.model.train_cfg = None
        model = build_detector(self.cfg.model, test_cfg=self.cfg.get('test_cfg'))
        if load != "latest":
            load = f"epoch_{load}"

        checkpoint = load_checkpoint(model, f"{self.cfg.work_dir}/{load}.pth", map_location='cpu')
        model.CLASSES = checkpoint['meta']['CLASSES']
        return model
