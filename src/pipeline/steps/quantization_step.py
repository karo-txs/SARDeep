from src.interfaces.quantization import Quantization
from src.infra.configs.config import Configuration
from src.pipeline.utils.loader import Loader
from dataclasses import dataclass, field
from src.interfaces.step import Step
from src.pipeline.functions import *
from typing import List
import json


@dataclass
class Quantization(Step):
    model: dict = field(default=None)
    load_epoch: str = field(default="latest")
    approach_names: list = field(default_factory=lambda: [])
    approachs: List[Quantization] = field(default_factory=lambda: [])

    def init_approachs(self):
        config = Configuration(self.model)
        cfg = config.load_config_for_test("voc")

        loader = Loader(cfg)
        dataset, data_loader = loader.load_dataset()

        model = loader.load_model(self.load_epoch)

        for approach in self.approach_names:
            if approach == "PytorchDynamic":
                self.approachs.append(
                    PytorchDynamicQuantization(model=model, model_path=cfg.work_dir,
                                               dataloader=data_loader,
                                               dataset=dataset,
                                               model_dict=self.model))
            elif approach == "PytorchStatic":
                self.approachs.append(
                    PytorchStaticQuantization(model=model, model_path=cfg.work_dir,
                                              dataloader=data_loader,
                                              dataset=dataset,
                                              model_dict=self.model))
        config_paths = dict(quantizations=[])
        for approach in self.approachs:
            config_paths["quantizations"].append(dict(path=approach.quantized_path,
                                                      test_path=f"/quantization/{approach.base_path}",
                                                      model_save_name=approach.model_save_name))

        with open(f"{cfg.work_dir}/config_paths.json", "w") as jsonFile:
            json.dump(config_paths, jsonFile)

    def run_step(self):
        self.init_approachs()
        for approach in self.approachs:
            approach.quantize()
            approach.test_quantized_model(eval_types=["voc", "coco"], )
