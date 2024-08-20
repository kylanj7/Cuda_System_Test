import torch
import time
import psutil
import os
from datetime import datetime
import statistics
import pynvml
import cpuinfo
import subprocess
import re

MIN_GPU_TEMP = 65  # Minimum target GPU temperature in Celsius
MAX_GPU_TEMP = 79  # Maximum target GPU temperature in Celsius
ADJUSTMENT_FACTOR = 0.05  # Factor to adjust memory usage

class VRAMLearner:
    def __init__(self, total_memory):
        self.total_memory = total_memory
        self.max_safe_usage = 0.9  # Start with 90% as the maximum
        self.current_usage = 0.6  # Start with 60% usage
        self.error_thresholds = []  # Store memory usage levels that caused errors

    def determine_safe_vram_limit(self, handle):
        _, free = get_gpu_memory_info(handle)
        other_processes_usage = (self.total_memory - free) / self.total_memory

        # Adjust max_safe_usage based on other processes
        self.max_safe_usage = max(0.1, min(0.9, 1 - other_processes_usage - 0.1))  # Ensure it's between 10% and 90%

        # Ensure we don't exceed any learned error thresholds
        if self.error_thresholds:
            self.max_safe_usage = min(self.max_safe_usage, min(self.error_thresholds) * 0.95)

        # Ensure we have at least 1GB or 10% of total VRAM free, whichever is larger
        min_free = max(1 * 1024 * 1024 * 1024, self.total_memory * 0.1)
        self.max_safe_usage = max(0.1, min(self.max_safe_usage, (self.total_memory - min_free) / self.total_memory))

        # Set the current usage to 80% of our determined max safe usage
        self.current_usage = max(0.1, self.max_safe_usage * 0.8)

        return self.max_safe_usage, self.current_usage

    def adjust_usage(self, temp):
        if temp > MAX_GPU_TEMP:
            self.current_usage *= (1 - ADJUSTMENT_FACTOR)
        elif temp < MIN_GPU_TEMP:
            new_usage = self.current_usage * (1 + ADJUSTMENT_FACTOR)
            # Don't exceed max_safe_usage when increasing
            self.current_usage = min(new_usage, self.max_safe_usage)
        
        # Ensure we don't go below 40% of max_safe_usage
        self.current_usage = max(0.4 * self.max_safe_usage, self.current_usage)

    def record_error(self, error_usage):
        self.error_thresholds.append(error_usage)
        # Immediately reduce max_safe_usage
        self.max_safe_usage = min(self.max_safe_usage, error_usage * 0.9)
        self.current_usage = self.max_safe_usage * 0.8

def initialize_gpu():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    return handle

def get_gpu_temperature(handle):
    return pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

def get_gpu_memory_info(handle):
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return meminfo.total, meminfo.free

def calculate_max_matrix_size(available_memory):
    max_elements = max(0, int(available_memory * 0.95 / (3 * 4)))  # Ensure non-negative
    return max(1, int(max_elements ** 0.5))  # Ensure at least 1x1 matrix

def sustained_gpu_operation(seconds, handle, vram_learner):
    start_time = time.time()
    max_usage, current_usage = vram_learner.determine_safe_vram_limit(handle)
    print(f"Determined safe VRAM usage: {max_usage:.2%}, Starting at: {current_usage:.2%}")

    while time.time() - start_time < seconds:
        temp = get_gpu_temperature(handle)
        vram_learner.adjust_usage(temp)
        target_memory = max(0, int(vram_learner.total_memory * vram_learner.current_usage))
        max_matrix_size = calculate_max_matrix_size(target_memory)
        
        print(f"Current temperature: {temp}°C, Target memory usage: {vram_learner.current_usage:.2%}, Matrix size: {max_matrix_size}")
        
        try:
            if max_matrix_size < 1:
                raise ValueError("Matrix size too small, skipping this iteration")
            a = torch.randn(max_matrix_size, max_matrix_size, device='cuda')
            b = torch.randn(max_matrix_size, max_matrix_size, device='cuda')
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("Out of memory error. Adjusting limits.")
                torch.cuda.empty_cache()
                vram_learner.record_error(vram_learner.current_usage)
            else:
                raise e
        except ValueError as e:
            print(f"Calculation error: {e}. Adjusting limits.")
            vram_learner.record_error(vram_learner.current_usage)

def run_benchmark(device, handle, vram_learner):
    sustained_gpu_operation(30, handle, vram_learner)  # Run for 30 seconds
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent
    _, free_memory = get_gpu_memory_info(handle)
    gpu_memory_usage = (vram_learner.total_memory - free_memory) / vram_learner.total_memory * 100
    gpu_utilization = torch.cuda.utilization()
    gpu_temp = get_gpu_temperature(handle)
    return {
        "CPU Usage": cpu_usage,
        "RAM Usage": ram_usage,
        "GPU Memory Usage": gpu_memory_usage,
        "GPU Utilization": gpu_utilization,
        "GPU Temperature": gpu_temp
    }

def get_ram_info():
    try:
        result = subprocess.run(['sudo', 'dmidecode', '--type', '17'], capture_output=True, text=True)
        output = result.stdout
        speed = re.search(r'Speed: (\d+) MHz', output)
        speed = speed.group(1) if speed else "Unknown"
        capacity = psutil.virtual_memory().total / (1024**3)  # Convert to GB
        data_rate = re.search(r'Type: (DDR\d+)', output)
        data_rate = data_rate.group(1) if data_rate else "Unknown"
        return f"{capacity:.2f} GB, {speed} MHz, {data_rate}"
    except:
        return "Unable to retrieve detailed RAM information"

def get_cpu_info():
    info = cpuinfo.get_cpu_info()
    model = info.get('brand_raw', 'Unknown')
    cores = psutil.cpu_count(logical=False)
    threads = psutil.cpu_count(logical=True)
    return f"{model}, {cores} cores, {threads} threads"

def main():
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your PyTorch installation.")
        return
    
    device = torch.device("cuda")
    handle = initialize_gpu()
    total_memory, _ = get_gpu_memory_info(handle)
    total_memory_gb = total_memory / (1024**3)
    gpu_name = torch.cuda.get_device_name(0)
    
    print(f"Using GPU: {gpu_name}")
    print(f"Total VRAM: {total_memory_gb:.2f} GB")
    
    vram_learner = VRAMLearner(total_memory)
    num_runs = 20
    results = []
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(os.path.expanduser("~"), "Documents", f"AI Benchmark {gpu_name}.txt")
    
    with open(filename, "w") as f:
        f.write(f"AI Benchmark Results - {timestamp}\n")
        f.write(f"GPU: {gpu_name}\n")
        f.write(f"Total VRAM: {total_memory_gb:.2f} GB\n")
        f.write(f"CPU: {get_cpu_info()}\n")
        f.write(f"RAM: {get_ram_info()}\n")
        f.write(f"Number of runs: {num_runs}\n")
        f.write(f"Target GPU temperature range: {MIN_GPU_TEMP}°C - {MAX_GPU_TEMP}°C\n")
        f.write("VRAM usage: Dynamically adjusted based on system state and learning\n\n")
        
        for i in range(num_runs):
            print(f"Running benchmark {i+1}/{num_runs}")
            result = run_benchmark(device, handle, vram_learner)
            results.append(result)
            f.write(f"Run {i+1}:\n")
            for key, value in result.items():
                unit = "%" if "Usage" in key or "Utilization" in key else "°C" if "Temperature" in key else ""
                f.write(f"{key}: {value:.2f}{unit}\n")
            f.write(f"Current safe VRAM limit: {vram_learner.max_safe_usage:.2%}\n")
            f.write("\n")
        
        averages = {}
        for key in results[0].keys():
            values = [result[key] for result in results]
            avg = statistics.mean(values)
            std_dev = statistics.stdev(values)
            averages[key] = (avg, std_dev)
        
        f.write("\nFinal Average Results:\n")
        for key, (avg, std_dev) in averages.items():
            unit = "%" if "Usage" in key or "Utilization" in key else "°C" if "Temperature" in key else ""
            f.write(f"{key}: {avg:.2f} ± {std_dev:.2f}{unit}\n")
        
        f.write(f"\nFinal learned safe VRAM limit: {vram_learner.max_safe_usage:.2%}\n")
        if vram_learner.error_thresholds:
            f.write(f"Recorded error thresholds: {', '.join([f'{x:.2%}' for x in vram_learner.error_thresholds])}\n")
    
    pynvml.nvmlShutdown()
    print(f"\nBenchmark complete. Results saved to: {filename}")

if __name__ == "__main__":
    main()
