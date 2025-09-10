import psutil
import GPUtil
import numpy as np
import matplotlib.pyplot as plt
from time import time, sleep
from threading import Thread


class ResourceMonitor:
    """
    A simple class to monitor RAM, CPU, and GPU usage over time.

    Usage:
    ```python
    monitor = ResourceMonitor(interval=1)
    monitor.start()
    # Do some work
    monitor.stop()
    monitor.plot()
    ```
    """
    def __init__(self, interval=1):
        self.interval = interval
        self.cpu_usage = []
        self.ram_usage = []
        self.gpu_usage = {}
        self.timestamps = []
        self.running = False

    def start(self):
        self.running = True
        self.thread = Thread(target=self._monitor)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def _monitor(self):
        start_time = time()
        while self.running:
            self.timestamps.append(time() - start_time)
            self.cpu_usage.append(psutil.cpu_percent())
            self.ram_usage.append(psutil.virtual_memory().percent)
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                if gpu.id not in self.gpu_usage:
                    self.gpu_usage[gpu.id] = []
                self.gpu_usage[gpu.id].append(gpu.load * 100)
            sleep(self.interval)

    def plot(self):
        plt.figure(figsize=(14, 8))
        # Plot CPU usage
        plt.subplot(2, 1, 1)
        _plot_with_moving_mean(self.timestamps, self.cpu_usage, color='blue')
        plt.xlabel("Time (s)")
        plt.ylabel("CPU Usage (%)")
        plt.title("CPU Usage Over Time. Mean: {:.1f}%".format(sum(self.cpu_usage)/len(self.cpu_usage)))
        plt.grid(True)

        # Plot RAM usage
        plt.subplot(2, 1, 2)
        _plot_with_moving_mean(self.timestamps, self.ram_usage, color='orange')
        plt.xlabel("Time (s)")
        plt.ylabel("RAM Usage (%)")
        plt.title("RAM Usage Over Time. Mean: {:.1f}%".format(sum(self.ram_usage)/len(self.ram_usage)))
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot GPU usage for each GPU
        plt.figure(figsize=(14, 10))
        n_gpus = len(self.gpu_usage)
        for gpu_id, usage in self.gpu_usage.items():
            plt.subplot(n_gpus, 1, gpu_id + 1)
            _plot_with_moving_mean(self.timestamps, usage, color='green')
            plt.xlabel("Time (s)")
            plt.ylabel(f"GPU {gpu_id} Usage (%)")
            plt.title(f"GPU {gpu_id} Usage Over Time. Mean: {sum(usage)/len(usage):.1f}%")
            plt.grid(True)
        plt.tight_layout()
        plt.show()


def _plot_with_moving_mean(x, y, color, window_size=60):
    plt.plot(x, y, color=color, alpha=0.3)
    y_smooth = _moving_mean(np.array(y), window_size)
    plt.plot(x, y_smooth, color=color, linestyle='--')


def _moving_mean(a, window_size, mode="reflect"):
    # pad with edge values
    pad = window_size // 2
    a_padded = np.pad(a, pad_width=pad, mode=mode)
    return np.convolve(a_padded, np.ones(window_size), 'valid') / window_size
