import torch
import time

class LocalTimer:
    def __init__(self, device: torch.device):
        self.measurements = []
        self.start_time = None
        if device.type == "cuda":
            self.synchronize = lambda: torch.cuda.synchronize(device=device)
        if device.type == "cpu":
            self.synchronize = lambda: torch.cpu.synchronize(device=device)

    def __enter__(self): #executed when entering the context via "with"
        self.synchronize()
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if traceback is None:
            self.synchronize()
            end_time = time.time()
            elapsed_time = end_time - self.start_time
            self.measurements.append(elapsed_time)
        self.start_time = None
    def get_avg_elapsed_ms(self):
        return 1000 * (sum(self.measurements)/len(self.measurements))

    def reset(self):
        self.measurements = []
        self.start_time = None


