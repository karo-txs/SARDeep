from mmdet.utils import (get_device, replace_cfg_vals, update_data_root, compat_cfg, build_dp)
from mmdet.datasets import (build_dataloader, replace_ImageToTensor)
from mmcv.runner import (get_dist_info, load_checkpoint)
from src.infra.configs.config import Configuration
from dataclasses import dataclass, field
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
import os.path as osp
import argparse
import mmcv
import time


@dataclass
class Test:
    config_file: str
    epoch: int
    metrics: list = field(default_factory=lambda: ["mAP"])
    show_dir: str = field(default=None)
    out: str = field(default=None)
    show: bool = field(default=True)
    show_score_thr: float = field(default=0.3)
    configuration: Configuration = field(default=None)
    eval_options: list = field(default=None)

    def __post_init__(self):
        self.configuration = Configuration(base_file=self.config_file)
        self.cfg = self.configuration.cfg

        self.show_dir = f"{self.cfg.work_dir}/results"
        self.out = f"{self.cfg.work_dir}/results/results_epoch_{self.epoch}.pkl"

        self.cfg = replace_cfg_vals(self.cfg)

        # update data root according to MMDET_DATASETS
        update_data_root(self.cfg)
        self.cfg = compat_cfg(self.cfg)

        if 'pretrained' in self.cfg.model:
            self.cfg.model.pretrained = None
        elif 'init_cfg' in self.cfg.model.backbone:
            self.cfg.model.backbone.init_cfg = None

        if self.cfg.model.get('neck'):
            if isinstance(self.cfg.model.neck, list):
                for neck_cfg in self.cfg.model.neck:
                    if neck_cfg.get('rfp_backbone'):
                        if neck_cfg.rfp_backbone.get('pretrained'):
                            neck_cfg.rfp_backbone.pretrained = None
            elif self.cfg.model.neck.get('rfp_backbone'):
                if self.cfg.model.neck.rfp_backbone.get('pretrained'):
                    self.cfg.model.neck.rfp_backbone.pretrained = None

        self.cfg.device = get_device()

        test_dataloader_default_args = dict(
            samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)

        # in case the test dataset is concatenated
        if isinstance(self.cfg.data.test, dict):
            self.cfg.data.test.test_mode = True
            if self.cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
                # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                self.cfg.data.test.pipeline = replace_ImageToTensor(self.cfg.data.test.pipeline)
        elif isinstance(self.cfg.data.test, list):
            for ds_cfg in self.cfg.data.test:
                ds_cfg.test_mode = True
            if self.cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
                for ds_cfg in self.cfg.data.test:
                    ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

        test_loader_cfg = {
            **test_dataloader_default_args,
            **self.cfg.data.get('test_dataloader', {})
        }

        rank, _ = get_dist_info()

        # allows not to create
        if self.cfg.work_dir is not None and rank == 0:
            mmcv.mkdir_or_exist(osp.abspath(self.cfg.work_dir))
            timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            self.json_file = osp.join(f"{self.cfg.work_dir}/results", f'eval_{timestamp}.json')

        # build the dataloader
        self.dataset = build_dataset(self.cfg.data.test)
        data_loader = build_dataloader(self.dataset, **test_loader_cfg)

        # build the model and load checkpoint
        self.cfg.model.train_cfg = None
        model = build_detector(self.cfg.model, test_cfg=self.cfg.get('test_cfg'))

        checkpoint = load_checkpoint(model, f"{self.cfg.work_dir}/epoch_{self.epoch}.pth", map_location='cpu')

        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = self.dataset.CLASSES

        model = build_dp(model, self.cfg.device, device_ids=self.cfg.gpu_ids)
        self.outputs = single_gpu_test(model, data_loader, self.show, self.show_dir, self.show_score_thr)

    def test_model(self):
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
    parser.add_argument('-c', '--config-file',  help='configuration path name')
    parser.add_argument('-e', '--epoch', help='checkpoint epoch')
    args = vars(parser.parse_args())

    test = Test(config_file="config_fcos_sard_v1", epoch=1)
    test.test_model()
