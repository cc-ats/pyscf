# Author: Junjie Yang <yangjunjie0320@gmail.com> Zheng Pei and Yihan Shao
import time
import tempfile

from functools import reduce
import numpy
from numpy import dot, empty, complex128
import scipy

from pyscf import gto, scf
from pyscf import lib, lo
from pyscf.lib import logger
from pyscf.rt  import chkfile

from pyscf.rt.util import build_absorption_spectrum
from pyscf.rt.util import print_matrix, print_cx_matrix, errm, expm

from pyscf import __config__

# integration schemes
# unitary euler propagation
PRINT_MAT_NCOL    = getattr(__config__, 'rt_tdscf_print_mat_ncol',              7)
PC_TOL            = getattr(__config__, 'rt_tdscf_pc_tol',                  1e-14)
PC_MAX_ITER       = getattr(__config__, 'rt_tdscf_pc_max_iter',                20)

LAST      = 0 
LAST_HALF = 1
THIS      = 2
NEXT_HALF = 3
NEXT      = 4

def prop_step(rt_obj, dt, fock_prim, dm_prim):
    propogator = expm(-1j*dt*fock_prim)
    dm_prim_   = reduce(dot, [propogator, dm_prim, propogator.conj().T])
    dm_prim_   = (dm_prim_ + dm_prim_.conj().T)/2
    dm_ao_     = rt_obj.orth2ao_dm(dm_prim_)
    return dm_prim_, dm_ao_


class Propogator(lib.StreamObject):
    def __init__(self, rt_obj):
        self.rt_obj          = rt_obj
        self.verbose         = rt_obj.verbose

        self.temp_ts         = None
        self.temp_dm_ao      = None
        self.temp_dm_orth    = None
        self.temp_fock_orth  = None
        self.temp_fock_ao    = None

        self.step_dm_ao      = None
        self.step_dm_orth    = None
        self.step_fock_orth  = None
        self.step_fock_ao    = None

    def first_step(self, dm_ao_init, dm_orth_init, fock_ao_init, fock_orth_init):
        temp_matrix_num = [1]
        shape = dm_ao_init.shape

        self.temp_dm_ao     = empty(temp_matrix_num+shape, dtype=complex128)
        self.temp_dm_orth   = empty(temp_matrix_num+shape, dtype=complex128)
        self.temp_fock_ao   = empty(temp_matrix_num+shape, dtype=complex128)
        self.temp_fock_orth = empty(temp_matrix_num+shape, dtype=complex128)

        self.step_dm_ao     = empty(shape, dtype=complex128)
        self.step_dm_orth   = empty(shape, dtype=complex128)
        self.step_fock_ao   = empty(shape, dtype=complex128)
        self.step_fock_orth = empty(shape, dtype=complex128)

    def propagate_step(self, save_this_step=True, rt_result=rt_result):
        pass