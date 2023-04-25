from src.interfaces.step import Step
from dataclasses import dataclass


@dataclass
class DatasetPreparation(Step):
    def cross_validation(self):
        pass

    def run_step(self):
        pass
