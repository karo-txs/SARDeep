from dataclasses import dataclass, field
#from torcheval.metrics import Throughput
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
    #metric_throughput: Throughput = field(default=())
    time_monotonic: float = field(default=0.0)

    def start(self):
        #self.metric_throughput = Throughput()
        self.time_monotonic = time.monotonic()
        if self.device == "cpu":
            self.start_time = time.time()
        else:
            self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            self.starter.record()

    def finalize(self):
        if self.device == "cpu":
            end_time = time.time()
            curr_time = end_time - self.start_time
        else:
            self.ender.record()
            torch.cuda.synchronize()
            curr_time = self.starter.elapsed_time(self.ender) / 1000

        self.total_time += curr_time
        self.timings.append(curr_time)

    def calculate(self):
        elapsed_time_sec = time.monotonic() - self.time_monotonic
        #self.metric_throughput.update(len(self.timings), elapsed_time_sec)

        self.result = {"mean_syn": sum(self.timings) / len(self.timings),
                       "std_syn": statistics.stdev(self.timings),
                       "throughput": 0.0}
                       #"throughput": self.metric_throughput.compute().item()}
