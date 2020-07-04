# Author: Junjie Yang <yangjunjie0320@gmail.com> Zheng Pei and Yihan Shao
import time
import tempfile

from functools import reduce
import numpy
from numpy import dot, empty, complex128
import scipy

from pyscf import gto, scf
from pyscf import lib, lo
from pyscf.lib     import logger
from pyscf.rt      import chkfile
from pyscf.rt.util import errm

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
        self.verbose = verbose 
        
        self.rt_obj          = rt_obj
        self.step_size       = rt_obj.step_size

        self.step_obj         = None
        self.step_iter        = None
        self.temp_ts          = None
        self.temp_dm_aos      = None
        self.temp_dm_orths    = None
        self.temp_fock_orths  = None
        self.temp_fock_aos    = None

    def first_step(self, dm_ao_init, dm_orth_init, fock_ao_init, fock_orth_init,
                   step_obj=None, verbose=None):
        if step_obj is None:
            step_obj = self.step_obj
        if verbose is None:
            verbose = self.verbose

        self.temp_ts            =  numpy.array([0.0])
        self.temp_dm_aos        = [dm_ao_init]
        self.temp_dm_orths      = [dm_orth_init]
        self.temp_fock_aos      = [fock_ao_init]
        self.temp_fock_orths    = [fock_orth_init]
        self.step_iter = 0
        logger.info( self, '\nInitializing Propagator: %s', self.__class__)
        logger.debug(self, 'step_size = %f', self.step_size)
        logger.debug(self, 'step_iter=%d, t=%f', 0, 0.0)
        logger.info(self, '\n')

        return self.step_iter

    def propagate_step(self, step_obj=None, verbose=None):
        if step_obj is None:
            step_obj = self.step_obj
        if verbose is None:
            verbose = self.verbose

        cput_time = (time.clock(), time.time())
        next_dm_orth, next_dm_ao = self.rt_obj.propagate_step(
            self.step_size, self.temp_fock_orths[0], self.temp_dm_orths[0]
        )
        next_t         = self.temp_ts[0] + self.step_size
        next_iter_step = self.step_iter + 1

        next_hcore_ao  = self.rt_obj.get_hcore_ao(next_t)
        next_veff_ao   = self.rt_obj.get_veff_ao(dm_orth=next_dm_orth, dm_ao=next_dm_ao)
        next_fock_ao   = self.rt_obj.get_fock_ao(
            next_hcore_ao, dm_orth=next_dm_orth, dm_ao=next_dm_ao, veff_ao=next_veff_ao
            )
        next_fock_orth = self.rt_obj.get_fock_orth(
            next_hcore_ao, fock_ao=next_fock_ao, dm_orth=next_dm_orth, dm_ao=next_dm_ao, veff_ao=next_veff_ao
            )
        
        step_obj._update(
            next_t, next_iter_step, next_dm_ao, next_dm_orth,
            next_fock_ao, next_fock_orth, next_hcore_ao, next_veff_ao
            )

        self.temp_dm_aos[0]           = next_dm_ao
        self.temp_dm_orths[0]         = next_dm_orth
        self.temp_fock_aos[0]         = next_fock_ao
        self.temp_fock_orths[0]       = next_fock_orth

        self.step_iter  += 1
        self.temp_ts    += self.step_size
        logger.debug(self, 'step_iter=%d, t=%f au', next_iter_step, next_t)
        cput1 = logger.timer(self, 'propagate_step', *cput_time)

        return self.step_iter

class EulerPropogator(Propogator):
    pass

class MMUTPropogator(Propogator):
    def first_step(self, dm_ao_init, dm_orth_init, fock_ao_init, fock_orth_init,
                   step_obj=None, verbose=None):
        if step_obj is None:
            step_obj = self.step_obj
        if verbose is None:
            verbose = self.verbose

        self.temp_ts                  = numpy.array([0.0, self.step_size/2])

        next_half_t = self.step_size/2
        next_half_dm_orth, next_half_dm_ao = self.rt_obj.propagate_step(
            self.step_size/2, fock_orth_init, dm_orth_init
        )
        next_half_hcore_ao  = self.rt_obj.get_hcore_ao(next_half_t)
        next_half_veff_ao   = self.rt_obj.get_veff_ao(dm_orth=next_half_dm_orth, dm_ao=next_half_dm_ao)
        next_half_fock_ao   = self.rt_obj.get_fock_ao(
            next_half_hcore_ao, dm_orth=next_half_dm_orth, dm_ao=next_half_dm_ao, veff_ao=next_half_veff_ao
            )
        next_half_fock_orth = self.rt_obj.get_fock_orth(
            next_half_hcore_ao, fock_ao=next_half_fock_ao, dm_orth=next_half_dm_orth, dm_ao=next_half_dm_ao, veff_ao=next_half_veff_ao
            )

        self.temp_dm_aos           = [dm_ao_init     , next_half_dm_ao    ]
        self.temp_dm_orths         = [dm_orth_init   , next_half_dm_orth  ]
        self.temp_fock_aos         = [fock_ao_init   , next_half_fock_ao  ]
        self.temp_fock_orths       = [fock_orth_init , next_half_fock_orth]

        self.step_iter = 0
        logger.info( self, '\nInitializing Propagator: %s', self.__class__)
        logger.debug(self, 'step_size = %f', self.step_size)
        logger.debug(self, 'step_iter=%d, t=%f', 0, 0.0)
        logger.info(self, '\n')

        return self.step_iter

    def propagate_step(self, step_obj=None, verbose=None):
        if step_obj is None:
            step_obj = self.step_obj
        if verbose is None:
            verbose = self.verbose

        cput_time = (time.clock(), time.time())

        next_half_t = self.temp_ts[1] + self.step_size
        next_half_dm_orth, next_half_dm_ao = self.rt_obj.propagate_step(
            self.step_size, self.temp_fock_orths[0], self.temp_dm_orths[1]
        )
        next_half_hcore_ao  = self.rt_obj.get_hcore_ao(next_half_t)
        next_half_veff_ao   = self.rt_obj.get_veff_ao(dm_orth=next_half_dm_orth, dm_ao=next_half_dm_ao)
        next_half_fock_ao   = self.rt_obj.get_fock_ao(
            next_half_hcore_ao, dm_orth=next_half_dm_orth, dm_ao=next_half_dm_ao, veff_ao=next_half_veff_ao
            )
        next_half_fock_orth = self.rt_obj.get_fock_orth(
            next_half_hcore_ao, fock_ao=next_half_fock_ao, dm_orth=next_half_dm_orth, dm_ao=next_half_dm_ao, veff_ao=next_half_veff_ao
            )

        next_t = self.temp_ts[0] + self.step_size
        next_dm_orth, next_dm_ao = self.rt_obj.propagate_step(
            self.step_size, next_half_fock_orth, self.temp_dm_orths[0]
        )
        next_hcore_ao  = self.rt_obj.get_hcore_ao(next_t)
        next_veff_ao   = self.rt_obj.get_veff_ao(dm_orth=next_dm_orth, dm_ao=next_dm_ao)
        next_fock_ao   = self.rt_obj.get_fock_ao(
            next_hcore_ao, dm_orth=next_dm_orth, dm_ao=next_dm_ao, veff_ao=next_veff_ao
            )
        next_fock_orth = self.rt_obj.get_fock_orth(
            next_hcore_ao, fock_ao=next_fock_ao, dm_orth=next_dm_orth, dm_ao=next_dm_ao, veff_ao=next_veff_ao
            )            

        next_iter_step = self.step_iter + 1
        step_obj._update(
            next_t, next_iter_step, next_dm_ao, next_dm_orth,
            next_fock_ao, next_fock_orth, next_hcore_ao, next_veff_ao
            )

        self.temp_dm_aos           = [next_dm_ao     , next_half_dm_ao    ]
        self.temp_dm_orths         = [next_dm_orth   , next_half_dm_orth  ]
        self.temp_fock_aos         = [next_fock_ao   , next_half_fock_ao  ]
        self.temp_fock_orths       = [next_fock_orth , next_half_fock_orth]

        self.step_iter  += 1
        self.temp_ts    += self.step_size
        logger.debug(self, 'step_iter=%d, t=%f au', next_iter_step, next_t)
        cput1 = logger.timer(self, 'propagate_step', *cput_time)

        return self.step_iter

class EPPCPropogator(Propogator):
    def __init__(self, rt_obj, verbose=None, tol=None, max_iter=None):
        if verbose is None:
            verbose = rt_obj.verbose
        self.verbose = verbose 
        
        self.rt_obj          = rt_obj
        self.step_size       = rt_obj.step_size

        if tol is None:
            self.tol = PC_TOL
        else:
            self.tol = tol

        if max_iter is None:
            self.max_iter = PC_MAX_ITER
        else:
            self.max_iter = max_iter

        self.step_obj         = None
        self.step_iter        = None
        self.temp_ts          = None
        self.temp_dm_aos      = None
        self.temp_dm_orths    = None
        self.temp_fock_orths  = None
        self.temp_fock_aos    = None

    def propagate_step(self, step_obj=None, verbose=None):
        if step_obj is None:
            step_obj = self.step_obj
        if verbose is None:
            verbose = self.verbose

        cput_time = (time.clock(), time.time())

        step_converged      = False
        inner_iter          = 0

        next_t         = self.temp_ts[0] + self.step_size
        next_iter_step = self.step_iter + 1
        next_hcore_ao  = self.rt_obj.get_hcore_ao(next_t)

        next_dm_orth_p, next_dm_ao_p = self.rt_obj.propagate_step(
            self.step_size, self.temp_fock_orths[0], self.temp_dm_orths[0]
        )
        while (not step_converged) and inner_iter <= self.max_iter:
            inner_iter += 1
            next_veff_ao_p   = self.rt_obj.get_veff_ao(dm_orth=next_dm_orth_p, dm_ao=next_dm_ao_p)
            next_fock_ao_p   = self.rt_obj.get_fock_ao(
                next_hcore_ao, dm_orth=next_dm_orth_p, dm_ao=next_dm_ao_p, veff_ao=next_veff_ao_p
                )
            next_fock_orth_p = self.rt_obj.get_fock_orth(
                next_hcore_ao, fock_ao=next_fock_ao_p, dm_orth=next_dm_orth_p, dm_ao=next_dm_ao_p, veff_ao=next_veff_ao_p
                )

            next_half_fock_orth = (next_fock_orth_p + self.temp_fock_orths[0])/2
            next_dm_orth_c, next_dm_ao_c = self.rt_obj.propagate_step(
                self.step_size, next_half_fock_orth, self.temp_dm_orths[0]
            )
            err = errm(next_dm_orth_c, next_dm_orth_p)
            step_converged = (err<self.tol)
            next_dm_orth_p, next_dm_ao_p = next_dm_orth_c, next_dm_ao_c
        
        if not step_converged:
            logger.warn(self, "PC step is not converged, step_iter = %d, inner_iter = %d, err = %e (tol = %e)", 
            self.step_iter, inner_iter, err, self.tol)
        step_obj._update(
            next_t, next_iter_step, next_dm_ao_p, next_dm_orth_p,
            next_fock_ao_p, next_fock_orth_p, next_hcore_ao, next_veff_ao_p
            )

        self.temp_dm_aos[0]           = next_dm_ao_p
        self.temp_dm_orths[0]         = next_dm_orth_p
        self.temp_fock_aos[0]         = next_fock_ao_p
        self.temp_fock_orths[0]       = next_fock_orth_p

        self.step_iter  += 1
        self.temp_ts    += self.step_size
        logger.debug(self, 'step_iter=%d, t=%f au', next_iter_step, next_t)
        logger.debug(self, 'PC step: inner_iter = %d, err = %e (tol = %e)', inner_iter, err, self.tol)
        cput1 = logger.timer(self, 'propagate_step', *cput_time)

        return self.step_iter

class LFLPPCPropogator(Propogator):
    def __init__(self, rt_obj, verbose=None, tol=None, max_iter=None):
        if verbose is None:
            verbose = rt_obj.verbose
        self.verbose = verbose 
        
        self.rt_obj          = rt_obj
        self.step_size       = rt_obj.step_size

        if tol is None:
            self.tol = PC_TOL
        else:
            self.tol = tol

        if max_iter is None:
            self.max_iter = PC_MAX_ITER
        else:
            self.max_iter = max_iter

        self.step_obj         = None
        self.step_iter        = None
        self.temp_ts          = None
        self.temp_dm_aos      = None
        self.temp_dm_orths    = None
        self.temp_fock_orths  = None
        self.temp_fock_aos    = None

    def first_step(self, dm_ao_init, dm_orth_init, fock_ao_init, fock_orth_init,
                   step_obj=None, verbose=None):
        if step_obj is None:
            step_obj = self.step_obj
        if verbose is None:
            verbose = self.verbose

        self.temp_ts                  = numpy.array([0.0, self.step_size/2])

        next_half_t = self.step_size/2
        next_half_dm_orth, next_half_dm_ao = self.rt_obj.propagate_step(
            self.step_size/2, fock_orth_init, dm_orth_init
        )
        next_half_hcore_ao  = self.rt_obj.get_hcore_ao(next_half_t)
        next_half_veff_ao   = self.rt_obj.get_veff_ao(dm_orth=next_half_dm_orth, dm_ao=next_half_dm_ao)
        next_half_fock_ao   = self.rt_obj.get_fock_ao(
            next_half_hcore_ao, dm_orth=next_half_dm_orth, dm_ao=next_half_dm_ao, veff_ao=next_half_veff_ao
            )
        next_half_fock_orth = self.rt_obj.get_fock_orth(
            next_half_hcore_ao, fock_ao=next_half_fock_ao, dm_orth=next_half_dm_orth, dm_ao=next_half_dm_ao, veff_ao=next_half_veff_ao
            )

        self.temp_dm_aos           = [dm_ao_init   ]
        self.temp_dm_orths         = [dm_orth_init ]
        self.temp_fock_aos         = [fock_ao_init ]
        self.temp_fock_orths       = [fock_orth_init , next_half_fock_orth]

        self.step_iter = 0
        logger.info( self, '\nInitializing Propagator: %s', self.__class__)
        logger.debug(self, 'step_size = %f', self.step_size)
        logger.debug(self, 'step_iter=%d, t=%f', 0, 0.0)
        logger.info(self, '\n')

        return self.step_iter

    def propagate_step(self, step_obj=None, verbose=None):
        if step_obj is None:
            step_obj = self.step_obj
        if verbose is None:
            verbose = self.verbose

        cput_time = (time.clock(), time.time())

        step_converged      = False
        inner_iter          = 0

        next_t         = self.temp_ts[0] + self.step_size
        next_iter_step = self.step_iter + 1
        next_hcore_ao  = self.rt_obj.get_hcore_ao(next_t)

        next_half_fock_orth_p = 2*self.temp_fock_orths[1] - self.temp_fock_orths[0]
        while (not step_converged) and inner_iter <= self.max_iter:
            inner_iter += 1
            next_dm_orth, next_dm_ao = self.rt_obj.propagate_step(
            self.step_size, next_half_fock_orth_p, self.temp_dm_orths[0]
            )
            
            next_veff_ao   = self.rt_obj.get_veff_ao(dm_orth=next_dm_orth, dm_ao=next_dm_ao)
            next_fock_ao   = self.rt_obj.get_fock_ao(
                next_hcore_ao, dm_orth=next_dm_orth, dm_ao=next_dm_ao, veff_ao=next_veff_ao
                )
            next_fock_orth = self.rt_obj.get_fock_orth(
                next_hcore_ao, fock_ao=next_fock_ao, dm_orth=next_dm_orth, dm_ao=next_dm_ao, veff_ao=next_veff_ao
                )

            next_half_fock_orth_c = (next_fock_orth + self.temp_fock_orths[0])/2
            err = errm(next_half_fock_orth_p, next_half_fock_orth_c)
            step_converged = (err<self.tol)
            next_half_fock_orth_p = next_half_fock_orth_c
        
        if not step_converged:
            logger.warn(self, "PC step is not converged, step_iter = %d, inner_iter = %d, err = %e (tol = %e)", 
            self.step_iter, inner_iter, err, self.tol)
        step_obj._update(
            next_t, next_iter_step, next_dm_ao, next_dm_orth,
            next_fock_ao, next_fock_orth, next_hcore_ao, next_veff_ao
            )

        self.temp_dm_aos           = [next_dm_ao   ]
        self.temp_dm_orths         = [next_dm_orth ]
        self.temp_fock_aos         = [next_fock_ao ]
        self.temp_fock_orths       = [next_fock_orth , next_half_fock_orth_p]

        self.step_iter  += 1
        self.temp_ts    += self.step_size
        logger.debug(self, 'step_iter=%d, t=%f au', next_iter_step, next_t)
        logger.debug(self, 'PC step: inner_iter = %d, err = %e (tol = %e)', inner_iter, err, self.tol)
        cput1 = logger.timer(self, 'propagate_step', *cput_time)

        return self.step_iter