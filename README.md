# Tensorflor GPU Test Script
This script tests correct usage of GPU from the Tensorflow library.

## Creating The Environment
To create the environment, run the poetry installation command:
```bash
poetry install
```
And then activate the virtual environment in the current shell:
```bash
poetry shell
```
Finally run the `gpu_test.py` script:
```bash
python3 gpu_test.py
```

## Example Output
Some Example output tested on an Asus G15 with the following specifications:
- Ubuntu 22.04
- Kernel 5.18.0
- Python 3.9
- CUDA 11.7
- cudNN 8.4.1

The resulting output:
```bash
tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 6124 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3070 Laptop GPU, pci bus id: 0000:01:00.0, compute capability: 8.6
2022-10-01 18:07:14.243139: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 2023497728 exceeds 10% of free system memory.
2022-10-01 18:07:14.576248: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 2023497728 exceeds 10% of free system memory.
2022-10-01 18:07:15.764786: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8401
Time (s) to convolve 32x7x7x3 filter over random 100x100x100x3 images (batch x height x width x channel). Sum of ten runs.
CPU (s):
2022-10-01 18:07:16.685553: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 2023497728 exceeds 10% of free system memory.
2022-10-01 18:07:17.014557: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 2023497728 exceeds 10% of free system memory.
2022-10-01 18:07:17.535343: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 2023497728 exceeds 10% of free system memory.
8.451914049999687
GPU (s):
0.3170320769995669
GPU speedup over CPU: 26x

```
## Benchmark
