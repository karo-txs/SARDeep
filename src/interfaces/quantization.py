from src.infra.configs.config import Configuration
from src.pipeline.steps.test_step import Test
from src.pipeline.utils.loader import Loader
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import os.path as osp
from torch import nn
import mmcv
import json


@dataclass
class Quantization(ABC):
    model: nn.Module
    dataloader: any
    dataset: any
    model_dict: dict
    model_path: str
    model_save_name: str = field(default="")
    base_path: str = field(default="")
    quantized_model: nn.Module = field(default=None)
    quantized_path: str = field(default=None)

    def make_dirs(self):
        mmcv.mkdir_or_exist(osp.abspath(f"{self.model_path}/quantization/{self.base_path}"))

    @abstractmethod
    def quantize(self):
        pass

    def test_quantized_model(self, eval_types: list, export_results: bool = True):
        test = Test(model=self.model_dict)
        config = Configuration(self.model_dict)

        for eval_type in eval_types:
            cfg = config.load_config_for_test(eval_type)

            loader = Loader(cfg)
            dataset, data_loader = loader.load_dataset()

            self.quantized_model.CLASSES = ("person",)

            data_test = config.base_file["datasets"]["paths"][config.base_file["datasets"]["dataset_type"]]["test"][
                "name"]
            show_dir = f"""{cfg.work_dir}/test_{config.device}/{data_test}/quantization/{self.base_path}"""

            out = f"""{cfg.work_dir}/test_{config.device}/{data_test}/quantization/{self.base_path}/results_{eval_type}.pkl"""

            test.test_model(model=self.quantized_model,
                            cfg=cfg,
                            config=config,
                            data_loader=data_loader,
                            dataset=dataset,
                            show_dir=show_dir,
                            out=out,
                            data_test=data_test,
                            eval_type=eval_type,
                            export_results=export_results)

    def upload_config(self):
        config = Configuration(self.model_dict)
        data_test = config.base_file["datasets"]["paths"][config.base_file["datasets"]["dataset_type"]]["test"][
            "name"]

        config_dict = dict(is_quantized=True, approach=self.base_path)
        mmcv.mkdir_or_exist(osp.abspath(f"{self.model_path}/test_{config.device}/{data_test}/quantization/{self.base_path}/"))
        with open(f"{self.model_path}/test_{config.device}/{data_test}/quantization/{self.base_path}/config.json", "w") as jsonFile:
            json.dump(config_dict, jsonFile)
