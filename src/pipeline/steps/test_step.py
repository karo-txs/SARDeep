from src.pipeline.functions.dataset.dataset_loader import DatasetLoader
from mmcv.runner import (get_dist_info, load_checkpoint)
from src.infra.configs.config import Configuration
from dataclasses import dataclass, field
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from src.interfaces.step import Step
from mmdet.utils import build_dp
from typing import List
import os.path as osp
import argparse
import mmcv
import time


@dataclass
class Test(Step):
    models: List[dict] = field(default=None)
    metrics: list = field(default_factory=lambda: ["mAP"])
    show: bool = field(default=True)
    show_score_thr: float = field(default=0.3)

    def run_step(self):

        for model_config in self.models:
            config = Configuration(model_config)
            cfg = config.load_config_for_test()

            show_dir = f"{cfg.work_dir}/results"
            out = f"{cfg.work_dir}/results/results.pkl"

            dataset, data_loader = DatasetLoader(cfg)

            rank, _ = get_dist_info()

            if cfg.work_dir is not None and rank == 0:
                mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
                timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
                json_file = osp.join(f"{cfg.work_dir}/results", f'eval_{timestamp}.json')

            # build the model and load checkpoint
            cfg.model.train_cfg = None
            model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))

            checkpoint = load_checkpoint(model, f"{cfg.work_dir}/latest.pth", map_location='cpu')

            if 'CLASSES' in checkpoint.get('meta', {}):
                model.CLASSES = checkpoint['meta']['CLASSES']
            else:
                model.CLASSES = dataset.CLASSES

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', help='configuration path name')
    parser.add_argument('-e', '--epoch', help='checkpoint epoch')
    args = vars(parser.parse_args())

    test = Test(config_file="config_faster_rcnn_sard_v1")
    test.run_step()
