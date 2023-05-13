from dataclasses import dataclass, field
from src.interfaces.step import Step
from src.pipeline.steps import *
from dotenv import load_dotenv
from typing import List
import json
import os

load_dotenv()


@dataclass
class PipelineBuilder:
    base_path: str = field(default=os.getenv("RESOURCE_PATH"))
    pipeline_file: str = field(default="pipeline")
    steps: List[Step] = field(default=None)

    def prepare_steps(self):
        self.steps = list()

        with open(f"{self.base_path}/{self.pipeline_file}.json") as f:
            pipeline_json = json.load(f)

        # Run priority steps
        for step in pipeline_json["priority_steps"]:
            if step["name"] == "DatasetPreparationStep":
                dataset_preparation = DatasetPreparation(**step)
                dataset_preparation.run_step()

        self._add_model_config(pipeline_json)

        for model in pipeline_json["models"]:
            for step in pipeline_json["steps"]:
                if step["name"] == "TrainDetectorStep":
                    self.steps.append(Train(model=model, **step))
                elif step["name"] == "TestDetectorStep":
                    self.steps.append(Test(model=model, **step))

    def _add_model_config(self, pipeline: dict):
        with open(f"{self.base_path}/models.json") as f:
            models_json = json.load(f)

        for idx, model_name in enumerate(pipeline["models"]):
            for model_dict in models_json["models"]:
                if f"""{model_dict["name"]}{model_dict["version"]}""" == model_name:
                    model_dict["datasets"] = self._get_dataset(pipeline["datasets"]["dataset_train"],
                                                               pipeline["datasets"]["fold"])
                    model_dict["work_dir"] = os.getenv("WORK_DIR")
                    model_dict["fine_tune"]["load_from"] = model_dict["fine_tune"]["load_from"].replace(
                        "CHECKPOINTS_PATH", os.getenv('CHECKPOINTS_PATH'))
                    pipeline["models"][idx] = model_dict

    def _get_dataset(self, name: str, fold: str) -> dict:
        with open(f"{self.base_path}/datasets.json") as f:
            datasets_json = json.load(f)

        for data_dict in datasets_json["datasets"]:
            if data_dict["name"] == name and data_dict["fold"] == fold:
                return data_dict
        return dict()

    def run_all(self):
        for step in self.steps:
            print(step.name)
            step.run_step()

    def run_by_step_name(self, step_name: str):
        for step in self.steps:
            print(step.name)
            if step_name in step.name:
                step.run_step()

    def reset(self):
        self.steps = []
