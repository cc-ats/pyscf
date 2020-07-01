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

class Propogator(lib.StreamObject):
    def __init__(self, rt_obj, verbose=None):
        if verbose is None:
            verbose = rt_obj.verbose
        
        self.rt_obj          = rt_obj
        self.dt              = rt_obj.dt

        self.rt_step          = None
        self.iter_step        = None
        self.temp_ts          = None
        self.temp_dm_aos      = None
        self.temp_dm_orths    = None
        self.temp_fock_orths  = None
        self.temp_fock_aos    = None

    def first_step(self, dm_ao_init, dm_orth_init, fock_ao_init, fock_orth_init,
                   rt_step=None, verbose=None):
        if rt_step is None:
            self.rt_step = rt_step
        if verbose is None:
            verbose = self.verbose

        temp_matrix_num = [1]
        shape = dm_ao_init.shape

        self.temp_ts        = numpy.array([0.0])
        self.temp_dm_aos     = empty(temp_matrix_num+shape, dtype=complex128)
        self.temp_dm_aos[0, :, :]     = dm_ao_init
        self.temp_dm_orths   = empty(temp_matrix_num+shape, dtype=complex128)
        self.temp_dm_orths[0, :, :]   = dm_orth_init
        self.temp_fock_aos   = empty(temp_matrix_num+shape, dtype=complex128)
        self.temp_fock_aos[0, :, :]   = fock_ao_init
        self.temp_fock_orths = empty(temp_matrix_num+shape, dtype=complex128)
        self.temp_fock_orths[0, :, :] = fock_orth_init

        self.iter_step = 0
        logger.info( self, '\nInitializing Propagator: %s', self.__class__)
        logger.debug(self, 'dt = %f', self.dt)
        logger.debug(self, 'iter_step=%d, t=%f', 0, 0.0)
        logger.info(logger, '\n')

        return self.iter_step

    def propagate_step(self, rt_step=None, verbose=None):
        if rt_step is None:
            self.rt_step = rt_step
        if verbose is None:
            verbose = self.verbose

        cput_time = (time.clock(), time.time())
        next_dm_orth, next_dm_ao = self.rt_obj.propagate_step(
            self.dt, self.temp_fock_orths[0], self.temp_dm_orths[0]
        )
        next_t         = self.temp_ts[0] + self.dt
        next_hcore_ao  = self.rt_obj.get_hcore_ao(next_t)
        next_veff_ao   = self.rt_obj.get_veff_ao(dm_orth=next_dm_orth, dm_ao=next_dm_ao)
        next_fock_ao   = self.rt_obj.get_fock_ao(
            next_hcore_ao, dm_orth=next_dm_orth, dm_ao=next_dm_ao, veff_ao=next_veff_ao
            )
        next_fock_orth = self.rt_obj.get_fock_orth(
            next_hcore_ao, fock_ao=next_fock_ao, dm_orth=next_dm_orth, dm_ao=next_dm_ao, veff_ao=next_veff_ao
            )
        next_iter_step = self.iter_step + 1
        rt_step.update_step(
            next_t, next_iter_step, next_dm_ao, next_dm_orth,
            next_fock_ao, next_fock_orth, next_hcore_ao, next_veff_ao
            )

        self.temp_dm_aos[0, :, :]     = next_dm_ao
        self.temp_dm_orths[0, :, :]   = next_dm_orth
        self.temp_fock_aos[0, :, :]   = next_fock_ao
        self.temp_fock_orths[0, :, :] = next_fock_orth

        self.iter_step  += 1
        self.temp_ts    += self.dt
        logger.debug(self, 'iter_step=%d, t=%f au', next_iter_step, next_t)
        cput1 = logger.timer(self, 'propagate_step', *cput_time)

        return self.iter_step

        

class EulerPropogator(Propogator):
    pass

class MMUTPropogator(Propogator):
    pass

class EPPCPropogator(Propogator):
    pass

class LFLPPCPropogator(Propogator):
    pass