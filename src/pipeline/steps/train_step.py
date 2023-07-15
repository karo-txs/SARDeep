from src.infra.configs.config import Configuration
from src.pipeline.utils.timer import Timer
from dataclasses import dataclass, field
from mmdet.models import build_detector
from mmdet.datasets import build_dataset
from mmdet.apis import train_detector
from src.interfaces.step import Step
from mmcv.utils import get_git_hash
from mmdet import __version__
import json


@dataclass
class Train(Step):
    model: dict = field(default=None)
    iteration: int = field(default=1)
    timer: Timer = field(default=Timer())

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

        self.timer.device = cfg_train["cfg"].device
        self.timer.start()
        train_detector(model,
                       datasets,
                       cfg_train["cfg"],
                       distributed=False,
                       validate=True,
                       timestamp=cfg_train["timestamp"],
                       meta=cfg_train["meta"])
        self.timer.finalize()
        self.timer.calculate()

        config_info = dict(is_quantized=False, approach=None,
                           train_time=self.timer.get_result_timer(),
                           iteration=self.iteration)

        with open(f"""{cfg_train["cfg"].work_dir}/config.json""", "w") as jsonFile:
            json.dump(config_info, jsonFile)
