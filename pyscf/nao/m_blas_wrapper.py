# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division
import numpy as np
from ctypes import POINTER, c_double, c_int, c_float
from pyscf.nao.m_sparsetools import libsparsetools


def spmv_wrapper(n, alpha, ap, x, beta = 0.0, incx = 1, incy = 1, lower=0):
  """ Small blas wrapper that is missing in scipy.linalg.blas in scipy.version<=1 """

  if ap.size != n*(n+1)//2:
    raise ValueError("simple wrapper, you MUST provide x.size = n, ap.size = n*(n+1)/2")
    
  if ap.dtype == np.float32:
    y = np.zeros((n), dtype=np.float32)
    libsparsetools.SSPMV_wrapper(c_int(lower), c_int(n), c_float(alpha),
                ap.ctypes.data_as(POINTER(c_float)),
                x.ctypes.data_as(POINTER(c_float)), c_int(incx), c_float(beta),
                y.ctypes.data_as(POINTER(c_float)), c_int(incy))
  elif ap.dtype == np.float64:
    y = np.zeros((n), dtype=np.float64)
    libsparsetools.DSPMV_wrapper(c_int(lower), c_int(n), c_double(alpha),
                ap.ctypes.data_as(POINTER(c_double)),
                x.ctypes.data_as(POINTER(c_double)), c_int(incx), c_double(beta),
                y.ctypes.data_as(POINTER(c_double)), c_int(incy))
  else:
    raise ValueError("dtype error, only np.float32 and np.float64 implemented")

  return y
