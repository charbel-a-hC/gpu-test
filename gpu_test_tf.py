import tensorflow as tf
import timeit

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')

CONV_OUT, KERNEL = 128, 7

def cpu():
  global CONV_OUT, KERNEL
  with tf.device('/cpu:0'):
    random_image_cpu = tf.random.normal((32, 500, 500, 3))
    net_cpu = tf.keras.layers.Conv2D(CONV_OUT, KERNEL)(random_image_cpu)
    return tf.math.reduce_sum(net_cpu)

def gpu():
  with tf.device('/device:GPU:0'):
    random_image_gpu = tf.random.normal((32, 500, 500, 3))
    net_gpu = tf.keras.layers.Conv2D(CONV_OUT, KERNEL)(random_image_gpu)
    return tf.math.reduce_sum(net_gpu)
  
# We run each op once to warm up; see: https://stackoverflow.com/a/45067900
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
