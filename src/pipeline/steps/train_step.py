from src.infra.configs.config import Configuration
from dataclasses import dataclass, field
from mmdet.models import build_detector
from mmdet.datasets import build_dataset
from mmdet.apis import train_detector
from src.interfaces.step import Step
from mmcv.utils import get_git_hash
from mmdet import __version__


@dataclass
class Train(Step):
    model: dict = field(default=None)

    def run_step(self):
        config = Configuration(self.model)
        cfg_train = config.load_config_for_train()

        model = build_detector(cfg_train["cfg"].model)
        model.init_weights()

        # Build dataset
        datasets = [build_dataset(cfg_train["cfg"].data.train)]

        model.CLASSES = datasets[0].CLASSES
        cfg_train["cfg"].checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)

        train_detector(model,
                       datasets,
                       cfg_train["cfg"],
                       distributed=False,
                       validate=True,
                       timestamp=cfg_train["timestamp"],
                       meta=cfg_train["meta"])
