# Author: Junjie Yang <yangjunjie0320@gmail.com> Zheng Pei and Yihan Shao
import time
import tempfile

from functools import reduce
import numpy
from numpy import dot
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

LAST      = 0 
LAST_HALF = 1
THIS      = 2
NEXT_HALF = 3
NEXT      = 4

def prop_step(tdscf, dt, fock_prim, dm_prim):
    propogator = expm(-1j*dt*fock_prim)
    dm_prim_   = reduce(dot, [propogator, dm_prim, propogator.conj().T])
    dm_prim_   = (dm_prim_ + dm_prim_.conj().T)/2
    dm_ao_     = tdscf.orth2ao_dm(dm_prim_)
    return dm_prim_, dm_ao_


def euler_prop(tdscf,  _temp_ts, _temp_dm_prims,   _temp_dm_aos,
               _temp_fock_prims, _temp_fock_aos):

    _temp_dm_prims[NEXT],   _temp_dm_aos[NEXT] = tdscf.prop_step(
        _temp_ts[NEXT] - _temp_ts[THIS],
        _temp_fock_prims[THIS], _temp_dm_prims[THIS],
        )

    _h1e_next_ao            = tdscf.get_hcore(t=_temp_ts[NEXT])
    _vhf_next_ao            = tdscf._scf.get_veff(dm=_temp_dm_aos[NEXT])
    _temp_fock_aos[NEXT]   = tdscf._scf.get_fock(dm=_temp_dm_aos[NEXT], h1e=_h1e_next_ao, vhf=_vhf_next_ao)
    ene_next_tot           = tdscf._scf.energy_tot(
        dm=_temp_dm_aos[NEXT], h1e=_h1e_next_ao, vhf=_vhf_next_ao
        )

    _temp_dm_prims[THIS]   = _temp_dm_prims[NEXT]
    _temp_dm_aos[THIS]     = _temp_dm_aos[NEXT]
    _temp_fock_prims[THIS] = tdscf.ao2orth_fock(_temp_fock_aos[NEXT])

    return ene_next_tot.real

# mmut propagation
def mmut_prop(tdscf,  _temp_ts, _temp_dm_prims,   _temp_dm_aos,
               _temp_fock_prims, _temp_fock_aos):

    _temp_dm_prims[NEXT_HALF],   _temp_dm_aos[NEXT_HALF] = tdscf.prop_step(
        _temp_ts[NEXT_HALF] - _temp_ts[LAST_HALF],
        _temp_fock_prims[THIS], _temp_dm_prims[LAST_HALF],
        )
    
    _temp_dm_prims[NEXT],   _temp_dm_aos[NEXT] = tdscf.prop_step(
        _temp_ts[NEXT] - _temp_ts[NEXT_HALF],
        _temp_fock_prims[THIS], _temp_dm_prims[NEXT_HALF],
        )

    _h1e_next_ao           = tdscf.get_hcore(t=_temp_ts[NEXT])
    _vhf_next_ao            = tdscf._scf.get_veff(dm=_temp_dm_aos[NEXT])
    _temp_fock_aos[NEXT]   = tdscf._scf.get_fock(dm=_temp_dm_aos[NEXT], h1e=_h1e_next_ao, vhf=_vhf_next_ao)
    ene_next_tot           = tdscf._scf.energy_tot(dm=_temp_dm_aos[NEXT], h1e=_h1e_next_ao, vhf=_vhf_next_ao)

    _temp_dm_prims[THIS]   = _temp_dm_prims[NEXT]
    _temp_dm_aos[THIS]     = _temp_dm_aos[NEXT]
    _temp_fock_prims[THIS] = tdscf.ao2orth_fock(_temp_fock_aos[NEXT])
    _temp_fock_aos[THIS]     = _temp_fock_aos[NEXT]

    _temp_dm_prims[LAST_HALF]   = _temp_dm_prims[NEXT_HALF]
    _temp_dm_aos[LAST_HALF]     = _temp_dm_aos[NEXT_HALF]

    return ene_next_tot.real

# rkmk1 propagation
def rkmk1_prop(tdscf,  _temp_ts, _temp_dm_prims,   _temp_dm_aos,
               _temp_fock_prims, _temp_fock_aos):

    _temp_dm_prims[NEXT_HALF],   _temp_dm_aos[NEXT_HALF] = tdscf.prop_step(
        _temp_ts[NEXT_HALF] - _temp_ts[THIS],
        _temp_fock_prims[THIS], _temp_dm_prims[THIS],
        )

    _h1e_next_half_ao            = tdscf.get_hcore(t=_temp_ts[NEXT_HALF])
    _vhf_next_half_ao            = tdscf._scf.get_veff(dm=_temp_dm_aos[NEXT_HALF])
    _temp_fock_aos[NEXT_HALF]   = tdscf._scf.get_fock(dm=_temp_dm_aos[NEXT_HALF], h1e=_h1e_next_half_ao, vhf=_vhf_next_half_ao)
    _temp_fock_prims[NEXT_HALF] = tdscf.ao2orth_fock(_temp_fock_aos[NEXT_HALF])

    _temp_dm_prims[NEXT],   _temp_dm_aos[NEXT] = tdscf.prop_step(
        _temp_ts[NEXT] - _temp_ts[THIS],
        _temp_fock_prims[NEXT_HALF], _temp_dm_prims[THIS],
        )

    _h1e_next_ao            = tdscf.get_hcore(t=_temp_ts[NEXT])
    _vhf_next_ao            = tdscf._scf.get_veff(dm=_temp_dm_aos[NEXT])
    _temp_fock_aos[NEXT]   = tdscf._scf.get_fock(dm=_temp_dm_aos[NEXT], h1e=_h1e_next_ao, vhf=_vhf_next_ao)
    ene_next_tot           = tdscf._scf.energy_tot(dm=_temp_dm_aos[NEXT], h1e=_h1e_next_ao, vhf=_vhf_next_ao)

    _temp_dm_prims[THIS]   = _temp_dm_prims[NEXT]
    _temp_dm_aos[THIS]     = _temp_dm_aos[NEXT]
    _temp_fock_prims[THIS] = tdscf.ao2orth_fock(_temp_fock_aos[NEXT])

    return ene_next_tot.real

# amut1 propagation
def amut1_prop(tdscf,  _temp_ts, _temp_dm_prims,   _temp_dm_aos,
               _temp_fock_prims, _temp_fock_aos):

    _temp_dm_prims[NEXT_HALF],   _temp_dm_aos[NEXT_HALF] = tdscf.prop_step(
        _temp_ts[NEXT_HALF] - _temp_ts[THIS],
        _temp_fock_prims[THIS], _temp_dm_prims[THIS],
        )

    _h1e_next_half_ao            = tdscf.get_hcore(t=_temp_ts[NEXT_HALF])
    _vhf_next_half_ao            = tdscf._scf.get_veff(dm=_temp_dm_aos[NEXT_HALF])
    _temp_fock_aos[NEXT_HALF]   = tdscf._scf.get_fock(dm=_temp_dm_aos[NEXT_HALF], h1e=_h1e_next_half_ao, vhf=_vhf_next_half_ao)
    _temp_fock_prims[NEXT_HALF] = tdscf.ao2orth_fock(_temp_fock_aos[NEXT_HALF])

    _temp_dm_prims[NEXT],   _temp_dm_aos[NEXT] = tdscf.prop_step(
        _temp_ts[NEXT] - _temp_ts[THIS],
        _temp_fock_prims[NEXT_HALF], _temp_dm_prims[THIS],
        )

    _h1e_next_ao            = tdscf.get_hcore(t=_temp_ts[NEXT])
    _vhf_next_ao            = tdscf._scf.get_veff(dm=_temp_dm_aos[NEXT])
    _temp_fock_aos[NEXT]   = tdscf._scf.get_fock(dm=_temp_dm_aos[NEXT], h1e=_h1e_next_ao, vhf=_vhf_next_ao)
    ene_next_tot           = tdscf._scf.energy_tot(dm=_temp_dm_aos[NEXT], h1e=_h1e_next_ao, vhf=_vhf_next_ao)

    _temp_dm_prims[THIS]   = _temp_dm_prims[NEXT]
    _temp_dm_aos[THIS]     = _temp_dm_aos[NEXT]
    _temp_fock_prims[THIS] = tdscf.ao2orth_fock(_temp_fock_aos[NEXT])

    return ene_next_tot.real

# amut2 propagation
def amut2_prop(tdscf,  _temp_ts, _temp_dm_prims,   _temp_dm_aos,
               _temp_fock_prims, _temp_fock_aos):

    _temp_dm_prims[NEXT_HALF],   _temp_dm_aos[NEXT_HALF] = tdscf.prop_step(
        _temp_ts[NEXT_HALF] - _temp_ts[THIS],
        _temp_fock_prims[THIS], _temp_dm_prims[THIS],
        )

    _h1e_next_half_ao            = tdscf.get_hcore(t=_temp_ts[NEXT_HALF])
    _vhf_next_half_ao            = tdscf._scf.get_veff(dm=_temp_dm_aos[NEXT_HALF])
    _temp_fock_aos[NEXT_HALF]   = tdscf._scf.get_fock(dm=_temp_dm_aos[NEXT_HALF], h1e=_h1e_next_half_ao, vhf=_vhf_next_half_ao)
    _temp_fock_prims[NEXT_HALF] = tdscf.ao2orth_fock(_temp_fock_aos[NEXT_HALF])


    _temp_dm_prims[NEXT_HALF],   _temp_dm_aos[NEXT_HALF] = tdscf.prop_step(
        _temp_ts[NEXT_HALF] - _temp_ts[THIS],
        _temp_fock_prims[NEXT_HALF], _temp_dm_prims[THIS],
        )

    _h1e_next_half_ao            = tdscf.get_hcore(t=_temp_ts[NEXT_HALF])
    _vhf_next_half_ao            = tdscf._scf.get_veff(dm=_temp_dm_aos[NEXT_HALF])
    _temp_fock_aos[NEXT_HALF]   = tdscf._scf.get_fock(dm=_temp_dm_aos[NEXT_HALF], h1e=_h1e_next_half_ao, vhf=_vhf_next_half_ao)
    _temp_fock_prims[NEXT_HALF] = tdscf.ao2orth_fock(_temp_fock_aos[NEXT_HALF])
    
    _temp_dm_prims[NEXT],   _temp_dm_aos[NEXT] = tdscf.prop_step(
        _temp_ts[NEXT] - _temp_ts[THIS],
        _temp_fock_prims[NEXT_HALF], _temp_dm_prims[THIS],
        )

    _h1e_next_ao            = tdscf.get_hcore(t=_temp_ts[NEXT])
    _vhf_next_ao            = tdscf._scf.get_veff(dm=_temp_dm_aos[NEXT])
    _temp_fock_aos[NEXT]   = tdscf._scf.get_fock(dm=_temp_dm_aos[NEXT], h1e=_h1e_next_ao, vhf=_vhf_next_ao)
    ene_next_tot           = tdscf._scf.energy_tot(dm=_temp_dm_aos[NEXT], h1e=_h1e_next_ao, vhf=_vhf_next_ao)

    _temp_dm_prims[THIS]   = _temp_dm_prims[NEXT]
    _temp_dm_aos[THIS]     = _temp_dm_aos[NEXT]
    _temp_fock_prims[THIS] = tdscf.ao2orth_fock(_temp_fock_aos[NEXT])
    
    return ene_next_tot.real

# amut3 propagation
def amut3_prop(tdscf,  _temp_ts, _temp_dm_prims,   _temp_dm_aos,
               _temp_fock_prims, _temp_fock_aos):

    _temp_dm_prims[NEXT_HALF],   _temp_dm_aos[NEXT_HALF] = tdscf.prop_step(
        _temp_ts[NEXT_HALF] - _temp_ts[THIS],
        _temp_fock_prims[THIS], _temp_dm_prims[THIS],
        )

    _h1e_next_half_ao            = tdscf.get_hcore(t=_temp_ts[NEXT_HALF])
    _vhf_next_half_ao            = tdscf._scf.get_veff(dm=_temp_dm_aos[NEXT_HALF])
    _temp_fock_aos[NEXT_HALF]   = tdscf._scf.get_fock(dm=_temp_dm_aos[NEXT_HALF], h1e=_h1e_next_half_ao, vhf=_vhf_next_half_ao)
    _temp_fock_prims[NEXT_HALF] = tdscf.ao2orth_fock(_temp_fock_aos[NEXT_HALF])


    _temp_dm_prims[NEXT_HALF],   _temp_dm_aos[NEXT_HALF] = tdscf.prop_step(
        _temp_ts[NEXT_HALF] - _temp_ts[THIS],
        _temp_fock_prims[NEXT_HALF], _temp_dm_prims[THIS],
        )

    _h1e_next_half_ao            = tdscf.get_hcore(t=_temp_ts[NEXT_HALF])
    _vhf_next_half_ao            = tdscf._scf.get_veff(dm=_temp_dm_aos[NEXT_HALF])
    _temp_fock_aos[NEXT_HALF]   = tdscf._scf.get_fock(dm=_temp_dm_aos[NEXT_HALF], h1e=_h1e_next_half_ao, vhf=_vhf_next_half_ao)
    _temp_fock_prims[NEXT_HALF] = tdscf.ao2orth_fock(_temp_fock_aos[NEXT_HALF])


    _temp_dm_prims[NEXT_HALF],   _temp_dm_aos[NEXT_HALF] = tdscf.prop_step(
        _temp_ts[NEXT_HALF] - _temp_ts[THIS],
        _temp_fock_prims[NEXT_HALF], _temp_dm_prims[THIS],
        )

    _h1e_next_half_ao            = tdscf.get_hcore(t=_temp_ts[NEXT_HALF])
    _vhf_next_half_ao            = tdscf._scf.get_veff(dm=_temp_dm_aos[NEXT_HALF])
    _temp_fock_aos[NEXT_HALF]   = tdscf._scf.get_fock(dm=_temp_dm_aos[NEXT_HALF], h1e=_h1e_next_half_ao, vhf=_vhf_next_half_ao)
    _temp_fock_prims[NEXT_HALF] = tdscf.ao2orth_fock(_temp_fock_aos[NEXT_HALF])
    
    _temp_dm_prims[NEXT],   _temp_dm_aos[NEXT] = tdscf.prop_step(
        _temp_ts[NEXT] - _temp_ts[THIS],
        _temp_fock_prims[NEXT_HALF], _temp_dm_prims[THIS],
        )

    _h1e_next_ao            = tdscf.get_hcore(t=_temp_ts[NEXT])
    _vhf_next_ao            = tdscf._scf.get_veff(dm=_temp_dm_aos[NEXT])
    _temp_fock_aos[NEXT]   = tdscf._scf.get_fock(dm=_temp_dm_aos[NEXT], h1e=_h1e_next_ao, vhf=_vhf_next_ao)
    ene_next_tot           = tdscf._scf.energy_tot(dm=_temp_dm_aos[NEXT], h1e=_h1e_next_ao, vhf=_vhf_next_ao)

    _temp_dm_prims[THIS]   = _temp_dm_prims[NEXT]
    _temp_dm_aos[THIS]     = _temp_dm_aos[NEXT]
    _temp_fock_prims[THIS] = tdscf.ao2orth_fock(_temp_fock_aos[NEXT])
    
    return ene_next_tot.real

# lflp-pc propagation
def lflp_pc_prop(tdscf,  _temp_ts, _temp_dm_prims,   _temp_dm_aos,
                 _temp_fock_prims, _temp_fock_aos, tol=PC_TOL, max_iter=PC_MAX_ITER):

    step_converged      = False
    inner_iter          = 0

    _h1e_next_half_ao      = tdscf.get_hcore(t=_temp_ts[NEXT_HALF])
    _h1e_next_ao           = tdscf.get_hcore(t=_temp_ts[NEXT])

    _fock_prim_next_half_p = 2*_temp_fock_prims[THIS] - _temp_fock_prims[LAST_HALF]
    while (not step_converged) and inner_iter <= max_iter:
        inner_iter += 1
        _temp_dm_prims[NEXT], _temp_dm_aos[NEXT]  = tdscf.prop_step(
            _temp_ts[NEXT] - _temp_ts[THIS], _fock_prim_next_half_p, _temp_dm_prims[THIS]
        )

        _dm_ao_next_half_c     = (_temp_dm_aos[NEXT] + _temp_dm_aos[THIS])/2
        _vhf_ao_next_half      = tdscf._scf.get_veff(dm=_dm_ao_next_half_c)
        _fock_ao_next_half_c   = tdscf._scf.get_fock(dm=_dm_ao_next_half_c, h1e=_h1e_next_half_ao, vhf=_vhf_ao_next_half)
        _fock_prim_next_half_c = tdscf.ao2orth_fock(_fock_ao_next_half_c)
        
        err = errm(_fock_prim_next_half_p, _fock_prim_next_half_c)
        logger.debug(tdscf, "inner_iter = %d, err = %e", inner_iter, err)
        if inner_iter >= 3:
            logger.info(tdscf, "inner_iter = %d, err = %e", inner_iter, err)
        step_converged = (err<tol)
        _fock_prim_next_half_p = _fock_prim_next_half_c
    
    _vhf_ao_next           = tdscf._scf.get_veff(dm=_temp_dm_aos[NEXT])
    _temp_fock_aos[NEXT]   = tdscf._scf.get_fock(dm=_temp_dm_aos[NEXT], h1e=_h1e_next_ao, vhf=_vhf_ao_next)
    _temp_fock_prims[NEXT] = tdscf.ao2orth_fock(_temp_fock_aos[NEXT])

    _temp_dm_prims[THIS]     = _temp_dm_prims[NEXT]
    _temp_dm_aos[THIS]       = _temp_dm_aos[NEXT]
    _temp_fock_aos[THIS]     = _temp_fock_aos[NEXT]
    _temp_fock_prims[THIS]   = _temp_fock_prims[NEXT]

    _temp_fock_prims[LAST_HALF] = _fock_prim_next_half_p
    return tdscf._scf.energy_tot(dm=_temp_dm_aos[NEXT], h1e=_h1e_next_ao, vhf=_vhf_ao_next).real

# ep-pc propagation
def ep_pc_prop(tdscf,  _temp_ts, _temp_dm_prims,   _temp_dm_aos,
                 _temp_fock_prims, _temp_fock_aos, tol=PC_TOL, max_iter=PC_MAX_ITER):

    step_converged      = False
    inner_iter          = 0
    _h1e_next_ao         = tdscf.get_hcore(t=_temp_ts[NEXT])

    _dm_prim_next_p, _dm_ao_next_p  = tdscf.prop_step(
            _temp_ts[NEXT] - _temp_ts[THIS], _temp_fock_prims[THIS], _temp_dm_prims[THIS]
        )
    while (not step_converged) and inner_iter <= max_iter:
        inner_iter += 1

        _vhf_ao_next                = tdscf._scf.get_veff(dm=_dm_ao_next_p)
        _temp_fock_aos[NEXT]        = tdscf._scf.get_fock(dm=_dm_ao_next_p, h1e=_h1e_next_ao, vhf=_vhf_ao_next)
        _temp_fock_prims[NEXT]      = tdscf.ao2orth_fock(_temp_fock_aos[NEXT])
        _temp_fock_prim_next_half   = (_temp_fock_prims[NEXT]+ _temp_fock_prims[THIS])/2

        _dm_prim_next_c, _dm_ao_next_c  = tdscf.prop_step(
            _temp_ts[NEXT] - _temp_ts[THIS], _temp_fock_prim_next_half, _temp_dm_prims[THIS]
        )

        err = errm(_dm_prim_next_c, _dm_prim_next_p)
        logger.debug(tdscf, "inner_iter = %d, err = %e", inner_iter, err)
        if inner_iter >= 3:
            logger.info(tdscf, "inner_iter = %d, err = %e", inner_iter, err)
        step_converged = (err<tol)
        _dm_prim_next_p = _dm_prim_next_c
        _dm_ao_next_p   = _dm_ao_next_c

    _temp_dm_prims[THIS]     = _dm_prim_next_c
    _temp_dm_aos[THIS]       = _dm_ao_next_c
    _temp_fock_aos[THIS]     = _temp_fock_aos[NEXT]
    _temp_fock_prims[THIS]   = _temp_fock_prims[NEXT]
    
    return tdscf._scf.energy_tot(dm=_dm_ao_next_c, h1e=_h1e_next_ao, vhf=_vhf_ao_next).real

# amut-pc propagation
def amut_pc_prop(tdscf,  _temp_ts, _temp_dm_prims,   _temp_dm_aos,
                 _temp_fock_prims, _temp_fock_aos, tol=PC_TOL, max_iter=PC_MAX_ITER):

    step_converged      = False
    inner_iter          = 0
    _h1e_next_ao         = tdscf.get_hcore(t=_temp_ts[NEXT])
    _h1e_next_half_ao    = tdscf.get_hcore(t=_temp_ts[NEXT_HALF])

    _dm_prim_next_half_p, _dm_ao_next_half_p  = tdscf.prop_step(
            _temp_ts[NEXT_HALF] - _temp_ts[THIS], _temp_fock_prims[THIS], _temp_dm_prims[THIS]
        )
    while (not step_converged) and inner_iter <= max_iter:
        inner_iter += 1

        _vhf_ao_next_half                = tdscf._scf.get_veff(dm=_dm_ao_next_half_p)
        _temp_fock_aos[NEXT_HALF]        = tdscf._scf.get_fock(dm=_dm_ao_next_half_p, h1e=_h1e_next_half_ao, vhf=_vhf_ao_next_half)
        _temp_fock_prims[NEXT_HALF]      = tdscf.ao2orth_fock(_temp_fock_aos[NEXT_HALF])

        _dm_prim_next_half_c, _dm_ao_next_half_c  = tdscf.prop_step(
            _temp_ts[NEXT_HALF] - _temp_ts[THIS], _temp_fock_prims[NEXT_HALF], _temp_dm_prims[THIS]
        )

        err = errm(_dm_prim_next_half_c, _dm_prim_next_half_p)
        logger.debug(tdscf, "inner_iter = %d, err = %e", inner_iter, err)
        if inner_iter >= 3:
            logger.info(tdscf, "inner_iter = %d, err = %e", inner_iter, err)
        step_converged = (err<tol)
        _dm_prim_next_half_p = _dm_prim_next_half_c
        _dm_ao_next_half_p   = _dm_ao_next_half_c

    _temp_dm_prims[NEXT], _temp_dm_aos[NEXT]  = tdscf.prop_step(
            _temp_ts[NEXT] - _temp_ts[THIS], _temp_fock_prims[NEXT_HALF], _temp_dm_prims[THIS]
        )
    _vhf_ao_next                = tdscf._scf.get_veff(dm=_temp_dm_aos[NEXT])
    _temp_fock_aos[NEXT]        = tdscf._scf.get_fock(dm=_temp_dm_aos[NEXT], h1e=_h1e_next_ao, vhf=_vhf_ao_next)
    _temp_fock_prims[NEXT]      = tdscf.ao2orth_fock(_temp_fock_aos[NEXT])

    _temp_dm_prims[THIS]     = _temp_dm_prims[NEXT]
    _temp_dm_aos[THIS]       = _temp_dm_aos[NEXT]
    _temp_fock_aos[THIS]     = _temp_fock_aos[NEXT]
    _temp_fock_prims[THIS]   = _temp_fock_prims[NEXT]

    return tdscf._scf.energy_tot(dm=_temp_dm_aos[NEXT], h1e=_h1e_next_ao, vhf=_vhf_ao_next).real