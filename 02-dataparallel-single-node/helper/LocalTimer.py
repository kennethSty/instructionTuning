import time
import torch

class LocalTimer:
    def __init__(self, device: torch.device):
        self.__synchronize = self.__set_synchronize(device)
        self.__measurements = []

    def __set_synchronize(device: torch.device):
        """
        Synchronizes all threads within a given device. Important to determine a clear starttime for on-device processes.
        Scope is for the provided device only, ensuring other devices are not idle due to synchronizing one device.
        """
        if device.type == "cuda":
            return lambda: torch.cuda.synchronize(device=device)
        if else:
            return lambda: torch.cpu.synchronize(device=device)

    def __enter__(self):
        self.__synchronize()
        self.__start_time = time.time()

    def __exit__(self):
        # Only measure time if within "with" statement no exception occured
        if traceback is None: 
            self.__synchronize()
            end_time = time.time()
            elapsed_time = self.start_time - end_time
            self.__measurements.append(elapsed_time)

        self.__start_time = None

    def get_avg_elapsed_ms(self):
        return 1000 * (sum(self.__measurements) / len(self.__measurements))

    def reset(self):
        self.__measurements = []
        self.__start_time = None
