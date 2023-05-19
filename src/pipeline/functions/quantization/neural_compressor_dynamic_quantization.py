from neural_compressor.config import PostTrainingQuantConfig, AccuracyCriterion
from src.infra.configs.config import Configuration
from src.interfaces.quantization import Quantization
from mmdet.apis import single_gpu_test
from neural_compressor import quantization
from mmdet.utils import build_dp
from dataclasses import dataclass
import torch


@dataclass
class NeuralCompressorDynamicQuantization(Quantization):

    def __post_init__(self):
        self.base_path = "neural_compressor/dynamic"
        self.make_dirs()

    def eval_fn(self, model):
        config = Configuration(self.model_dict)
        cfg = config.load_config_for_test()

        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, self.dataloader)
        eval_kwargs = cfg.get('evaluation', {}).copy()

        for key in [
            'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
            'rule', 'dynamic_intervals'
        ]:
            eval_kwargs.pop(key, None)

        eval_kwargs.update(dict(metric="mAP"))

        metric = self.dataset.evaluate(outputs, **eval_kwargs)
        return metric["AP50"]

    def quantize(self):
        acc = AccuracyCriterion(tolerable_loss=0.05)
        conf = PostTrainingQuantConfig(approach="dynamic", accuracy_criterion=acc)

        self.quantized_model = quantization.fit(model=self.model,
                                                conf=conf,
                                                calib_dataloader=self.dataloader,
                                                eval_func=self.eval_fn)
        self.quantized_model.save(f"{self.model_path}/quantization/{self.base_path}")
        self.quantized_model = self.quantized_model.model
        self.upload_config()
