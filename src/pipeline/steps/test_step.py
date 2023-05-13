from src.infra.configs.config import Configuration
from dataclasses import dataclass, field
from mmcv.runner import get_dist_info
from mmdet.apis import single_gpu_test
from src.pipeline.functions import *
from src.interfaces.step import Step
from mmdet.utils import build_dp
import os.path as osp
import mmcv
import time


@dataclass
class Test(Step):
    model: dict = field(default=None)
    metrics: list = field(default_factory=lambda: ["mAP"])
    show: bool = field(default=True)
    show_score_thr: float = field(default=0.3)

    def run_step(self):
        config = Configuration(self.model)
        cfg = config.load_config_for_test()

        data_test = config.base_file["datasets"]["paths"]["test"]["name"]
        show_dir = f"""{cfg.work_dir}/test/{data_test}"""
        out = f"""{cfg.work_dir}/test/{data_test}/results.pkl"""

        loader = Loader(cfg)
        dataset, data_loader = loader.load_dataset()

        rank, _ = get_dist_info()

        if cfg.work_dir is not None and rank == 0:
            mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
            timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            json_file = osp.join(f"{cfg.work_dir}/test/{data_test}", f'eval_{timestamp}.json')

        model = loader.load_model()
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, data_loader, self.show, show_dir, self.show_score_thr)

        print(f'\nwriting results to {out}')
        mmcv.dump(outputs, out)

        for metric_name in self.metrics:
            eval_kwargs = cfg.get('evaluation', {}).copy()

            for key in [
                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule', 'dynamic_intervals'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=metric_name))

            metric = dataset.evaluate(outputs, **eval_kwargs)
            metric_dict = dict(config=config.config_file, metric=metric)
            mmcv.dump(metric_dict, json_file)
