from src.infra.configs.config import Configuration
from dataclasses import dataclass, field
from mmdet.models import build_detector
from mmdet.datasets import build_dataset
from mmdet.apis import train_detector
from src.interfaces.step import Step
from mmcv.utils import get_git_hash
from mmdet import __version__
import argparse


@dataclass
class Train(Step):
    config_file: str = field(default=None)

    def __post_init__(self):
        self.step_name = "TrainDetector"
        config = Configuration(self.config_file)
        self.cfg_train = config.load_config_for_train()

    def run_step(self):
        model = build_detector(self.cfg_train["cfg"].model)
        model.init_weights()

        # Build dataset
        datasets = [build_dataset(self.cfg_train["cfg"].data.train)]

        model.CLASSES = datasets[0].CLASSES
        self.cfg_train["cfg"].checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)

        train_detector(model,
                       datasets,
                       self.cfg_train["cfg"],
                       distributed=False,
                       validate=True,
                       timestamp=self.cfg_train["timestamp"],
                       meta=self.cfg_train["meta"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', help='configuration path name')
    args = vars(parser.parse_args())

    train = Train(config_file="config_ssd_sard_v1")
    train.run_step()
