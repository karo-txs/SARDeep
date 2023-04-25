from src.pipeline.functions.dataset.dataset_loader import DatasetLoader
from mmcv.runner import (get_dist_info, load_checkpoint)
from src.infra.configs.config import Configuration
from dataclasses import dataclass, field
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from src.interfaces.step import Step
from mmdet.utils import build_dp
import os.path as osp
import argparse
import mmcv
import time


@dataclass
class Test(Step):
    config_file: str = field(default=None)
    epoch: int = field(default=20)
    metrics: list = field(default_factory=lambda: ["mAP"])
    show_dir: str = field(default=None)
    out: str = field(default=None)
    show: bool = field(default=True)
    show_score_thr: float = field(default=0.3)
    configuration: Configuration = field(default=None)
    eval_options: list = field(default=None)

    def __post_init__(self):
        self.step_name = "TestDetector"
        self.configuration = Configuration(base_file=self.config_file)
        self.cfg = self.configuration.load_config_for_test()

        self.show_dir = f"{self.cfg.work_dir}/results"
        self.out = f"{self.cfg.work_dir}/results/results_epoch_{self.epoch}.pkl"

        self.dataset, self.data_loader = DatasetLoader(self.cfg)

        rank, _ = get_dist_info()

        if self.cfg.work_dir is not None and rank == 0:
            mmcv.mkdir_or_exist(osp.abspath(self.cfg.work_dir))
            timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            self.json_file = osp.join(f"{self.cfg.work_dir}/results", f'eval_{timestamp}.json')

        # build the model and load checkpoint
        self.cfg.model.train_cfg = None
        model = build_detector(self.cfg.model, test_cfg=self.cfg.get('test_cfg'))

        checkpoint = load_checkpoint(model, f"{self.cfg.work_dir}/epoch_{self.epoch}.pth", map_location='cpu')

        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = self.dataset.CLASSES

        model = build_dp(model, self.cfg.device, device_ids=self.cfg.gpu_ids)
        self.outputs = single_gpu_test(model, self.data_loader, self.show, self.show_dir, self.show_score_thr)

    def run_step(self):
        print(f'\nwriting results to {self.out}')
        mmcv.dump(self.outputs, self.out)

        kwargs = {} if self.eval_options is None else self.eval_options

        for metric_name in self.metrics:
            eval_kwargs = self.cfg.get('evaluation', {}).copy()

            for key in [
                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule', 'dynamic_intervals'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=metric_name, **kwargs))

            metric = self.dataset.evaluate(self.outputs, **eval_kwargs)
            metric_dict = dict(config=self.configuration.config_file, metric=metric)
            mmcv.dump(metric_dict, self.json_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', help='configuration path name')
    parser.add_argument('-e', '--epoch', help='checkpoint epoch')
    args = vars(parser.parse_args())

    test = Test(config_file="config_faster_rcnn_sard_v1", epoch=1)
    test.run_step()
