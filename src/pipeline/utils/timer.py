from dataclasses import dataclass, field
import statistics
import torch
import time


@dataclass
class Timer:
    device: str = field(default="gpu")
    batch_size: int = field(default=1)
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
            curr_time = end_time - self.start_time
        else:
            self.ender.record()
            torch.cuda.synchronize()
            curr_time = self.starter.elapsed_time(self.ender)

        self.total_time += curr_time
        self.timings.append(curr_time)

    def calculate(self):
        mean_syn = sum(self.timings) / len(self.timings)
        throughput = 60 / mean_syn

        self.result = {"mean_syn": mean_syn,
                       "std_syn": statistics.stdev(self.timings) if len(self.timings) > 1 else 0,
                       "throughput": throughput}

    def get_result_timer(self) -> str:
        sec = self.result["mean_syn"] % (24 * 3600)
        hour = sec // 3600
        sec %= 3600
        min = sec // 60
        sec %= 60
        print("seconds value in hours:", hour)
        print("seconds value in minutes:", min)
        print("%02d:%02d:%02d" % (hour, min, sec))

        return "%02d:%02d:%02d" % (hour, min, sec)
