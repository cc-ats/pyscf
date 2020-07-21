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
from pyscf.rt.util import errm

from pyscf.rt.util import print_matrix, print_cx_matrix, errm, expm
from pyscf.rt.util import expia_b_exp_ia

from pyscf import __config__

PC_TOL            = getattr(__config__, 'rt_tdscf_pc_tol',                   1e-8)
PC_MAX_ITER       = getattr(__config__, 'rt_tdscf_pc_max_iter',                20)

class Propogator(lib.StreamObject):
    r'''
    Propogator
    '''
    pass

class EulerPropogator(Propogator):
    r'''
     EulerPropogator

     Propagate MO density matrix/matricies using Euler method.
    '''
    def __init__(self, rt_obj, verbose=None):
        if verbose is None:
            verbose = rt_obj.verbose
        self.verbose = verbose 
        
        self.rt_obj          = rt_obj
        self.step_size       = None

        self.step_obj         = None
        self.step_iter        = None
        self.temp_ts          = None
        self.temp_dm_aos      = None
        self.temp_dm_orths    = None
        self.temp_fock_orths  = None
        self.temp_fock_aos    = None

        logger.info( self, '\nInitializing Propagator: %s', self.__class__)

    def first_step(self, dm_ao_init, dm_orth_init, fock_ao_init, fock_orth_init,
                   step_obj=None, verbose=None):
        r'''
        Prepare the propagator by setting the first step.

        The step would be saved in step_obj

        '''
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

        return self.step_iter

    def propagate_step(self, step_obj=None, verbose=None):
        r'''
        a) P'(t+dt) = e^{-i F'(t) dt} P'(t) e^{i F'(t) dt}

        All mats in MO basis

        '''
        if step_obj is None:
            step_obj = self.step_obj
        if verbose is None:
            verbose = self.verbose
        
        cput_time = (time.clock(), time.time())
        log = logger.new_logger(self, verbose)
        log.debug('\n    ----------------------------------------')
        next_dm_orth, next_dm_ao = self.rt_obj.propagate_step(
            self.step_size, self.temp_fock_orths[0], self.temp_dm_orths[0]
        )

        cput1          = log.timer('%20s'%"Propagate Step", *cput_time)
        next_t         = self.temp_ts[0] + self.step_size
        next_iter_step = self.step_iter + 1
        next_hcore_ao  = self.rt_obj.get_hcore_ao(next_t)

        cput2          = log.timer('%20s'%"Build hcore", *cput1)
        next_veff_ao   = self.rt_obj.get_veff_ao(dm_orth=next_dm_orth, dm_ao=next_dm_ao)
        next_fock_ao   = self.rt_obj.get_fock_ao(
            next_hcore_ao, dm_orth=next_dm_orth, dm_ao=next_dm_ao, veff_ao=next_veff_ao
            )
        next_fock_orth = self.rt_obj.get_fock_orth(
            next_hcore_ao, fock_ao=next_fock_ao, dm_orth=next_dm_orth, dm_ao=next_dm_ao, veff_ao=next_veff_ao
            )

        cput3 = log.timer('%20s'%"Build Fock", *cput2)
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
        cput4 = log.timer('%20s'%"Finalize", *cput3)
        cput5 = log.timer('%20s'%"Finish Step", *cput_time)
        logger.debug(self, '    step_iter=%d, t=%f au', next_iter_step, next_t)
        return self.step_iter

class MMUTPropogator(Propogator):
    r'''
     MMUTPropogator

     Propagate MO density matrix/matricies using
     modified-midpoint unitary transformation (MMUT)
     propagation algorithm.
    '''
    def __init__(self, rt_obj, verbose=None):
        if verbose is None:
            verbose = rt_obj.verbose
        self.verbose = verbose 
        
        self.rt_obj          = rt_obj
        self.step_size       = None

        self.step_obj         = None
        self.step_iter        = None
        self.temp_ts          = None
        self.temp_dm_aos      = None
        self.temp_dm_orths    = None
        self.temp_fock_orths  = None
        self.temp_fock_aos    = None

        logger.info( self, '\nInitializing Propagator: %s', self.__class__)

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

        return self.step_iter

    def propagate_step(self, step_obj=None, verbose=None):
        r'''


        All mats in MO basis

        '''
        if step_obj is None:
            step_obj = self.step_obj
        if verbose is None:
            verbose = self.verbose

        cput_time = (time.clock(), time.time())
        log = logger.new_logger(self, verbose)
        log.debug('\n    ########################################')

        next_t = self.temp_ts[0] + self.step_size
        next_dm_orth, next_dm_ao = self.rt_obj.propagate_step(
            self.step_size/2, self.temp_fock_orths[0], self.temp_dm_orths[1]
        )
        next_hcore_ao  = self.rt_obj.get_hcore_ao(next_t)
        next_veff_ao   = self.rt_obj.get_veff_ao(dm_orth=next_dm_orth, dm_ao=next_dm_ao)
        next_fock_ao   = self.rt_obj.get_fock_ao(
            next_hcore_ao, dm_orth=next_dm_orth, dm_ao=next_dm_ao, veff_ao=next_veff_ao
            )
        next_fock_orth = self.rt_obj.get_fock_orth(
            next_hcore_ao, fock_ao=next_fock_ao, dm_orth=next_dm_orth, dm_ao=next_dm_ao, veff_ao=next_veff_ao
            )

        next_half_t = self.temp_ts[1] + self.step_size
        next_half_dm_orth, next_half_dm_ao = self.rt_obj.propagate_step(
            self.step_size, next_fock_orth, self.temp_dm_orths[1]
        )

        next_iter_step = self.step_iter + 1
        step_obj._update(
            next_t, next_iter_step, next_dm_ao, next_dm_orth,
            next_fock_ao, next_fock_orth, next_hcore_ao, next_veff_ao
            )

        self.temp_dm_aos           = [next_dm_ao     , next_half_dm_ao    ]
        self.temp_dm_orths         = [next_dm_orth   , next_half_dm_orth  ]
        self.temp_fock_aos         = [next_fock_ao   , None               ]
        self.temp_fock_orths       = [next_fock_orth , None               ]

        self.step_iter  += 1
        self.temp_ts    += self.step_size

        log.debug('    step_iter=%d, t=%f au', next_iter_step, next_t)
        cput_time = log.timer('%20s'%'step finished', *cput_time)
        return self.step_iter

class PCPropogator(Propogator):
    pass

class EPPCPropogator(PCPropogator):
    def __init__(self, rt_obj, verbose=None, tol=None, max_iter=None):
        if verbose is None:
            verbose = rt_obj.verbose
        self.verbose = verbose 
        
        self.rt_obj          = rt_obj
        self.step_size       = None

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

        logger.info( self, '\nInitializing PC-Propagator: %s', self.__class__)
        logger.info( self, 'inner_max_iter = %d， inner_tol = %e', self.max_iter, self.tol)

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

        return self.step_iter

    def propagate_step(self, step_obj=None, verbose=None):
        '''      
        a) P'(t) -> P'(t+dt/2) using F'(t)
        b) F'(t+dt/2) << P'(t+dt/2)
        c) Propagate P'(t)->P'(t+dt) using new F'(t+1/2*dt) from b)
        d) F'(t+dt) << P'(t+dt)
        e) If F'(t+dt/2) == (F'(t+dt) + F'(t))/2, else go to c)
        '''
        if step_obj is None:
            step_obj = self.step_obj
        if verbose is None:
            verbose = self.verbose

        cput_time = (time.clock(), time.time())
        logger.debug(self, '\n    ########################################')

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
        cput1 = logger.timer(self, 'propagate_step', *cput_time)
        logger.debug(self, '    step_iter=%d, t=%f au', next_iter_step, next_t)
        logger.debug(self, '    PC step: inner_iter = %d, err = %e (tol = %e)', inner_iter, err, self.tol)
        
        return self.step_iter

class LFLPPCPropogator(PCPropogator):
    def __init__(self, rt_obj, verbose=None, tol=None, max_iter=None):
        if verbose is None:
            verbose = rt_obj.verbose
        self.verbose = verbose 
        
        self.rt_obj          = rt_obj
        self.step_size       = None

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

        logger.info( self, '\nInitializing PC-Propagator: %s', self.__class__)
        logger.info( self, 'inner_max_iter = %d， inner_tol = %e', self.max_iter, self.tol)

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

        return self.step_iter

    def propagate_step(self, step_obj=None, verbose=None):
        '''      
        a) Extrapolate F'(t+1/2*dt) from F'(t-1/2*dt) and F'(t)
        b) Propagate P'(t) -> P'(t+dt) using extrapolated F'(t+1/2*dt);
        c) F'(t+dt) << P'(t+dt)
        d) Propagate P'(t)->P'(t+dt) using new F'(t+1/2*dt) from c)
        e) If new P'(t+1/2*dt) is same as previous one exit, else go to (3)
        '''
        if step_obj is None:
            step_obj = self.step_obj
        if verbose is None:
            verbose = self.verbose

        cput_time = (time.clock(), time.time())
        logger.debug(self, '\n    ########################################')

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

        cput1 = logger.timer(self, 'propagate_step', *cput_time)
        logger.debug(self, '    step_iter=%d, t=%f au', next_iter_step, next_t)
        logger.debug(self, '    PC step: inner_iter = %d, err = %e (tol = %e)', inner_iter, err, self.tol)
        
        return self.step_iter
