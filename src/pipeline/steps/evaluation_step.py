import json

from src.infra.configs.config import Configuration
from dataclasses import dataclass, field
from src.interfaces.step import Step
import pandas as pd


@dataclass
class Evaluation(Step):
    model: dict = field(default=None)

    def run_step(self):
        config = Configuration(self.model)

        data_test = config.base_file["datasets"]["paths"][config.base_file["datasets"]["dataset_type"]]["test"]["name"]
        result_dir = f"""{config.cfg.work_dir}/test/{data_test}"""
        output_dir = f"""{config.cfg.work_dir}/eval/{data_test}"""

        with open(f"{result_dir}/datasets.json", "r") as jsonFile:
            data = json.load(jsonFile)

        results_voc = pd.read_pickle(f"{result_dir}/results_voc.pkl")
        results_coco = pd.read_pickle(f"{result_dir}/results_coco.pkl")

        result_metrics = {}




