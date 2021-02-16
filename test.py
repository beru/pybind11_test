#!/usr/bin/python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

import sys
sys.path.insert(0, 'build')
import pybind11_test

import time

# ret = pybind11_test.add(10, 20)
# print(ret)

# ret = pybind11_test.subtract(0, 10)
# print(ret)

class ScopeTime:
    def __init__(self, label):
        self.label = label
    def __enter__(self):
        self.start = time.perf_counter()
    def __exit__(self, exc_type, exc_value, traceback):
        end = time.perf_counter()
        elapsed_milliseconds = 1000 * (end - self.start)
        str = "{:<14} : {:.3f} ms".format(self.label, elapsed_milliseconds)
        print(str)


print(f"tf.executing_eagerly : {tf.executing_eagerly()}")

with ScopeTime("startup"):
    t0 = tf.zeros([1])
    del t0
print()

size = 1024 * 1024 # 1 million floats

for i in range(10):
    print("{:<14} : {} MB".format("size", 4 * size//(1024*1024)))
    with ScopeTime("tf.zeros"):
        t = tf.zeros([1024, 1024, size//(1024*1024)])

    # print(type(t))

    with ScopeTime("some opes"):
        t = tf.add(t, 10)
        t = tf.multiply(t, 4)
        t = tf.add(t, 10)
        t = tf.multiply(t, 4)
        t = tf.add(t, 10)
        t = tf.multiply(t, 4)
        t = tf.add(t, 10)
        t = tf.multiply(t, 4)
        tf.reduce_sum(t).numpy()  # sync...

    # print(t.shape)
    # print(t.device)

    with ScopeTime("to_dlpack"):
        packed = tf.experimental.dlpack.to_dlpack(t)

    with ScopeTime("add_gpu"):
        pybind11_test.add_gpu(packed, 1.1 + i)
        tf.reduce_sum(t).numpy() # sync...

    with ScopeTime("from_dlpack"):
        t = tf.experimental.dlpack.from_dlpack(packed)
    
    if 4 * tf.size(t).numpy() < 1024*1024*1024*2:
        with ScopeTime("numpy()"):
            arr = t.numpy()
    
    print(t[1023,1023,0])

    size *= 2


