from src.interfaces.step import Step
from typing import List


class PipelineBuilder:
    steps: List[Step]

    def prepare_steps(self):
        # load json pipeline and prepare steps
        pass

    def add_step(self, step: Step):
        self.steps.append(step)

    def run(self):
        for step in self.steps:
            step.run_step()

    def reset(self):
        self.steps = []
