from dataclasses import dataclass, field
from src.interfaces.step import Step
from src.pipeline.steps import *
from typing import List
import json


@dataclass
class PipelineBuilder:
    base_path: str = field(default="../../resource")
    pipeline_file: str = field(default="pipeline")
    steps: List[Step] = field(default=None)

    def prepare_steps(self):
        self.steps = list()

        with open(f"{self.base_path}/{self.pipeline_file}.json") as f:
            pipeline_json = json.load(f)

        self._add_model_config(pipeline_json)

        for model in pipeline_json["models"]:
            for step in pipeline_json["steps"]:
                if step["name"] == "TestDetectorStep":
                    self.steps.append(Test(model=model, **step))

                elif step["name"] == "TrainDetectorStep":
                    self.steps.append(Train(model=model, **step))

    def _add_model_config(self, pipeline: dict):
        with open(f"{self.base_path}/models.json") as f:
            models_json = json.load(f)

        for idx, model_name in enumerate(pipeline["models"]):
            for model_dict in models_json["models"]:
                if f"""{model_dict["name"]}{model_dict["version"]}""" == model_name:
                    self._add_dataset_config(model_dict, pipeline)
                    model_dict["work_dir"] = pipeline["work_dir"]
                    pipeline["models"][idx] = model_dict

    def _add_dataset_config(self, model_dict: dict, pipeline: dict):
        model_dict["datasets"] = dict()

        for dataset in pipeline["datasets"]:
            if "train" in dataset["use"]:
                model_dict["datasets"]["train"] = dataset["name"]

            if "test" in dataset["use"]:
                model_dict["datasets"]["test"] = dataset["name"]

            if "val" in dataset["use"]:
                model_dict["datasets"]["val"] = dataset["name"]

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
