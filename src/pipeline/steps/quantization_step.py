from src.interfaces.quantization import Quantization
from src.infra.configs.config import Configuration
from dataclasses import dataclass, field
from src.interfaces.step import Step
from src.pipeline.functions import *
from typing import List


@dataclass
class Quantization(Step):
    model: dict = field(default=None)
    approach_names: list = field(default_factory=lambda: [])
    approachs: List[Quantization] = field(default_factory=lambda: [])

    def __post_init__(self):
        config = Configuration(self.model)
        cfg = config.load_config_for_test("voc")

        loader = Loader(cfg)
        dataset, data_loader = loader.load_dataset()

        model = loader.load_model()

        for approach in self.approach_names:
            if approach == "PytorchDynamic":
                self.approachs.append(
                    PytorchDynamicQuantization(model=model, model_path=cfg.work_dir, dataloader=data_loader,
                                               dataset=dataset,
                                               model_dict=self.model))
            elif approach == "NeuralCompressorDynamic":
                self.approachs.append(
                    NeuralCompressorDynamicQuantization(model=model, model_path=cfg.work_dir, dataloader=data_loader,
                                                        dataset=dataset,
                                                        model_dict=self.model))
            elif approach == "NeuralCompressorStatic":
                self.approachs.append(
                    NeuralCompressorStaticQuantization(model=model, model_path=cfg.work_dir, dataloader=data_loader,
                                                       dataset=dataset,
                                                       model_dict=self.model))

    def run_step(self):
        for approach in self.approachs:
            approach.quantize()
            approach.test_quantized_model()
