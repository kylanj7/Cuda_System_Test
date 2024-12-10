# AI GPU Benchmark Tool

A sophisticated GPU benchmarking tool that uses PyTorch to evaluate GPU performance while implementing adaptive VRAM management and temperature monitoring. The tool learns optimal memory usage patterns and adjusts workloads based on real-time performance metrics.

## Features

- Adaptive VRAM usage learning
- Real-time temperature monitoring
- Dynamic workload adjustment
- Comprehensive system information gathering
- Detailed benchmark reporting
- Safe operation boundaries
- Multiple test iterations

## Prerequisites

- NVIDIA GPU with CUDA support
- Python 3.7+
- Required packages:
```bash
pip install torch pynvml psutil py-cpuinfo
```

## System Requirements

- NVIDIA GPU (CUDA compatible)
- Admin privileges for detailed RAM info
- Sufficient cooling system
- Windows/Linux OS

## Installation

1. Install required Python packages:
```bash
pip install -r requirements.txt
```
2. Ensure NVIDIA drivers are up to date
3. Verify CUDA installation

## Safety Features

- Temperature boundaries (65°C - 79°C)
- Adaptive VRAM usage
- Error threshold learning
- Automatic workload adjustment
- Memory safety margins

## Benchmarking Process

1. System Information Collection
   - GPU specifications
   - CPU details
   - RAM configuration
   - System temperatures

2. VRAM Management
   - Initial 60% usage
   - Dynamic adjustment
   - Error threshold learning
   - Safe limits enforcement

3. Performance Testing
   - Matrix multiplication operations
   - Temperature monitoring
   - Resource utilization tracking
   - Error handling

## Output

Generates a detailed report including:
- System specifications
- Average performance metrics
- Temperature readings
- Resource utilization
- VRAM usage patterns
- Standard deviations

## File Output

- Location: `~/Documents/AI Benchmark [GPU_NAME].txt`
- Format: Timestamped detailed report
- Contents: All benchmark iterations and averages

## Usage

Run the benchmark:
```python
python gpu_benchmark.py
```

## Monitoring

The tool monitors:
- GPU temperature
- VRAM usage
- CPU utilization
- System RAM usage
- GPU utilization

## Error Handling

- Out of memory protection
- Temperature safeguards
- System resource monitoring
- Graceful error recovery

## Limitations

- NVIDIA GPUs only
- Requires admin rights for full RAM info
- Temperature-dependent performance
- System-specific results

## Contributing

Feel free to submit issues and enhancement requests!

## License

[Specify your license here]

## Acknowledgments

- PyTorch team
- NVIDIA NVML library
- Python hardware monitoring tools
