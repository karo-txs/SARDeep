from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from torch import nn
from src.infra.configs.config import Configuration
from src.pipeline.steps import *


@dataclass
class Quantization(ABC):
    model: nn.Module
    dataloader: any
    dataset: any
    model_dict: dict
    model_path: str
    base_path: str = field(default="")
    quantized_model: nn.Module = field(default=None)

    @abstractmethod
    def quantize(self):
        pass

    def test_quantized_model(self):
        test = Test(model=self.model_dict)
        config = Configuration(self.model_dict)

        eval_types = ["voc", "coco"]

        for eval_type in eval_types:
            cfg = config.load_config_for_test(eval_type)

            data_test = config.base_file["datasets"]["paths"][config.base_file["datasets"]["dataset_type"]]["test"][
                "name"]
            show_dir = f"""{cfg.work_dir}/test/{data_test}/quantization"""

            out = f"""{cfg.work_dir}/test/{data_test}/results_{eval_type}.pkl"""

            test.test_model(model=self.quantized_model,
                            cfg=cfg,
                            config=config,
                            data_loader=self.dataloader,
                            dataset=self.dataset,
                            show_dir=show_dir,
                            out=out,
                            data_test=data_test,
                            eval_type=eval_type)
