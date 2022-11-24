import torch
import torch.nn

import timeit

gpu_available = torch.cuda.is_available()

if not gpu_available:
    raise SystemError("Cuda device not available")

CONV_OUT, KERNEL = 128, 7

def cpu():
    in_channels, batch_size, W, H = 3, 32, 500, 500
    random_image_cpu = torch.randn(size= (batch_size, in_channels, W, H)).cpu()
    conv_cpu = torch.nn.Conv2d(in_channels, 128, 4).cpu()
    out = conv_cpu(random_image_cpu)
    return torch.sum(out)

def gpu():
    in_channels, batch_size, W, H = 3, 32, 500, 500
    random_image_cuda = torch.randn(size= (batch_size, in_channels, W, H)).cuda()
    conv_gpu = torch.nn.Conv2d(in_channels, 128, 4).cuda()
    out = conv_gpu(random_image_cuda)
    return torch.sum(out)

# Warm up
cpu()
gpu()

# Run the op several times.
print(f'Time (s) to convolve {CONV_OUT}x{KERNEL}x{KERNEL}x3 filter over random 500x500x500x3 images '
      '(batch x height x width x channel). Sum of ten runs.')
print('CPU (s):')
cpu_time = timeit.timeit('cpu()', number=10, setup="from __main__ import cpu")
print(cpu_time)
print('GPU (s):')
gpu_time = timeit.timeit('gpu()', number=10, setup="from __main__ import gpu")
print(gpu_time)
print('GPU speedup over CPU: {}x'.format(int(cpu_time/gpu_time)))