from dataclasses import dataclass, field
import statistics
import torch
import time


@dataclass
class Timer:
    device: str = field(default="cpu")
    batch_size: int = field(default=64)
    timings: list = field(default_factory=lambda: [])
    result: dict = field(default=None)
    start_time: float = field(default=0.0)
    total_time: float = field(default=0.0)
    starter: torch.cuda.Event = field(default=None)
    ender: torch.cuda.Event = field(default=None)

    def start(self):
        if self.device == "cpu":
            self.start_time = time.time()
        else:
            self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            self.starter.record()

    def finalize(self):
        if self.device == "cpu":
            end_time = time.time()
            curr_time = self.start_time - end_time
        else:
            self.ender.record()
            torch.cuda.synchronize()
            curr_time = self.starter.elapsed_time(self.ender) / 1000

        self.total_time += curr_time
        self.timings.append(curr_time)

    def calculate(self):
        self.result = {"mean_syn": sum(self.timings) / len(self.timings),
                       "std_syn": statistics.stdev(self.timings),
                       "throughput": (len(self.timings) * self.batch_size) / self.total_time}
