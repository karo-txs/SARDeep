from mmdet.datasets import build_dataset, build_dataloader
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass, field
from typing import Tuple
from mmcv import Config


@dataclass
class DatasetLoader:
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
