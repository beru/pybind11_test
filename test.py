#!/usr/bin/python

import sys
sys.path.insert(0, 'build')
import pybind11_test

ret = pybind11_test.add(10, 20)
print(ret)

ret = pybind11_test.subtract(0, 10)
print(ret)

