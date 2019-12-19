import time
import sys
import tempfile

from functools import reduce

import numpy as np
import scipy
from scipy.fftpack import fft

from pyscf import gto, dft, df
from pyscf import lib, lo
from pyscf.lib import logger
from pyscf.rt  import chkfile

from pyscf import __config__

# TODO: chk file 
write = sys.stdout.write
MUTE_CHKFILE      = getattr(__config__, 'rt_tdscf_mute_chkfile', False)
DAMP_EXPO         = getattr(__config__, 'rt_tdscf_damp_expo',     1000)
PRINT_MAT_NCOL    = getattr(__config__, 'rt_tdscf_print_mat_ncol',   7)

def print_matrix(title, array_, ncols=7, fmt=' % 11.6e'):
    ''' printing a real rectangular matrix, or the real part of a complex matrix, ncols columns per batch '''
    array = array_.real
    write(title+'\n')
    m = array.shape[1]
    n = array.shape[0]
    #write('m=%d n=%d\n' % (m, n))
    nbatches = int(n/ncols)
    if nbatches * ncols < n: nbatches += 1
    for k in range(0, nbatches):
        write('     ')  
        j1 = ncols*k
        j2 = ncols*(k+1)
        if k == nbatches-1: j2 = n 
        for j in range(j1, j2):
            write('   %7d  ' % (j+1))
        write('\n')
        for i in range(0, m): 
            write(' %3d -' % (i+1))
            for j in range(j1, j2):
                write( fmt % array[j,i])
            write('\n')

def print_cx_matrix(title, cx_array_, ncols=7, fmt=' % 11.6e'):
    ''' printing a complex rectangular matrix, ncols columns per batch '''
    print_matrix(title+" Real Part ", cx_array_.real, ncols=ncols, fmt=fmt)
    print_matrix(title+" Imag Part ", cx_array_.imag, ncols=ncols, fmt=fmt)

def build_absorption_spectrum(tdscf, ndipole=None, damp_expo=DAMP_EXPO):
    if ndipole is None:
        ndipole = tdscf.ndipole

    ndipole_x = ndipole[:,0]
    ndipole_y = ndipole[:,1]
    ndipole_z = ndipole[:,2]
    
    ndipole_x = ndipole_x-ndipole_x[0]
    ndipole_y = ndipole_y-ndipole_y[0]
    ndipole_z = ndipole_z-ndipole_z[0]
    
    mw = 2.0 * np.pi * np.fft.fftfreq(
        tdscf.ntime.size, tdscf.dt
    )
    damp = np.exp(-tdscf.ntime/damp_expo)
    fwx = np.fft.fft(ndipole_x*damp)
    fwy = np.fft.fft(ndipole_y*damp)
    fwz = np.fft.fft(ndipole_z*damp)
    fw = (fwx.imag + fwy.imag + fwz.imag) / 3.0 
    sigma = - mw * fw
    mm = mw.size
    m  = mm//2

    mw = mw[:m]
    sigma = sigma[:m]
    scale = np.abs(sigma.max())
    return mw, sigma/scale

def merr(m1,m2):
    ''' check consistency '''
    n   = np.linalg.norm(m1-m2)
    r   = m1.shape[0]
    v   = np.linalg.eigvals(m1)
    vm  = v.max()
    e   = n/r/vm
    return np.abs(e)


def expm(m, do_bch=False):
    if not do_bch:
        return scipy.linalg.expm(m)
    else:
        raise NotImplementedError("BCH not implemented here")

def prop_step(tdscf, t_start, dm_prim, fock_prim, dt = None, build_fock = True):
    if dt == None:
        dt = tdscf.dt
    
    propogator = expm(-1j*dt*fock_prim)
    dm_prim_ = reduce(np.dot, [propogator, dm_prim, propogator.conj().T])

    dm_prim_   = (dm_prim_ + dm_prim_.conj().T)/2

    if tdscf.verbose >= logger.DEBUG1:
        print_cx_matrix("fock_prim"
        , fock_prim, ncols=PRINT_MAT_NCOL)
        print_cx_matrix("dm_prim"
        , dm_prim, ncols=PRINT_MAT_NCOL)
    dm_ao_     = orth2ao_dm(dm_prim_, tdscf.orth_xtuple)
    
    if build_fock:
        fock_ao_   = tdscf.mf.get_fock(dm=dm_ao_, h1e=tdscf.mf.get_hcore(t=(dt+t_start)))
        fock_prim_ = ao2orth_fock(fock_ao_, tdscf.orth_xtuple)
        return dm_prim_, dm_ao_, fock_prim_, fock_ao_
    else:
        return dm_prim_, dm_ao_

def euler_prop(tdscf,                  
               _temp_ts,         _temp_dm_prims,   _temp_dm_aos,
               _temp_fock_prims, _temp_fock_aos,        dt=None):
    if dt == None:
        dt = tdscf.dt
    _temp_dm_prims[4],   _temp_dm_aos[4],\
    _temp_fock_prims[4], _temp_fock_aos[4] = prop_step(
        tdscf, _temp_ts[2], _temp_dm_prims[2], _temp_fock_prims[2], dt = dt
        )
    _temp_dm_prims[2]   = _temp_dm_prims[4]
    _temp_dm_aos[2]     = _temp_dm_aos[4]
    _temp_fock_prims[2] = _temp_fock_prims[4]
    _temp_fock_aos[2]   = _temp_fock_aos[4]

def amut1_prop(tdscf,                  
               _temp_ts,         _temp_dm_prims,   _temp_dm_aos,
               _temp_fock_prims, _temp_fock_aos,        dt=None):
    if dt == None:
        dt = tdscf.dt

    _temp_dm_prims[3],   _temp_dm_aos[3],\
    _temp_fock_prims[3], _temp_fock_aos[3] = prop_step(
        tdscf, _temp_ts[2], _temp_dm_prims[2], _temp_fock_prims[2], dt = dt/2
        )
    
    _temp_dm_prims[4],   _temp_dm_aos[4],\
    _temp_fock_prims[4], _temp_fock_aos[4] = prop_step(
        tdscf, _temp_ts[2], _temp_dm_prims[2], _temp_fock_prims[3], dt = dt
        )

    _temp_dm_prims[2]   = _temp_dm_prims[4]
    _temp_dm_aos[2]     = _temp_dm_aos[4]
    _temp_fock_prims[2] = _temp_fock_prims[4]
    _temp_fock_aos[2]   = _temp_fock_aos[4]

def amut2_prop(tdscf,                  
               _temp_ts,         _temp_dm_prims,   _temp_dm_aos,
               _temp_fock_prims, _temp_fock_aos,        dt=None):
    if dt == None:
        dt = tdscf.dt

    _p_prim_2,   _p_ao_2,\
    _f_prim_2,   _f_ao_2  = prop_step(
        tdscf, _temp_ts[2], _temp_dm_prims[2], _temp_fock_prims[2], dt = dt/2
        )
    
    _p_prim_3,   _p_ao_3,\
    _f_prim_3,   _f_ao_3  = prop_step(
        tdscf, _temp_ts[2], _temp_dm_prims[2], _f_prim_2, dt = dt/2
        )

    _temp_dm_prims[4],   _temp_dm_aos[4],\
    _temp_fock_prims[4], _temp_fock_aos[4]  = prop_step(
        tdscf, _temp_ts[2], _temp_dm_prims[2], _f_prim_3, dt = dt
        )

    _temp_dm_prims[2]   = _temp_dm_prims[4]
    _temp_dm_aos[2]     = _temp_dm_aos[4]
    _temp_fock_prims[2] = _temp_fock_prims[4]
    _temp_fock_aos[2]   = _temp_fock_aos[4]

def amut3_prop(tdscf,                  
               _temp_ts,         _temp_dm_prims,   _temp_dm_aos,
               _temp_fock_prims, _temp_fock_aos,        dt=None):
    if dt == None:
        dt = tdscf.dt

    _p_prim_2,   _p_ao_2,\
    _f_prim_2,   _f_ao_2  = prop_step(
        tdscf, _temp_ts[2], _temp_dm_prims[2], _temp_fock_prims[2], dt = dt/2
        )
    
    _p_prim_3,   _p_ao_3,\
    _f_prim_3,   _f_ao_3  = prop_step(
        tdscf, _temp_ts[2], _temp_dm_prims[2], _f_prim_2, dt = dt/2
        )

    _p_prim_4,   _p_ao_4,\
    _f_prim_4,   _f_ao_4  = prop_step(
        tdscf, _temp_ts[2], _temp_dm_prims[2], _f_prim_3, dt = dt/2
        )

    _temp_dm_prims[4],   _temp_dm_aos[4],\
    _temp_fock_prims[4], _temp_fock_aos[4]  = prop_step(
        tdscf, _temp_ts[2], _temp_dm_prims[2], _f_prim_4, dt = dt
        )

    _temp_dm_prims[2]   = _temp_dm_prims[4]
    _temp_dm_aos[2]     = _temp_dm_aos[4]
    _temp_fock_prims[2] = _temp_fock_prims[4]
    _temp_fock_aos[2]   = _temp_fock_aos[4]

def aeut_prop(tdscf,                  
               _temp_ts,         _temp_dm_prims,   _temp_dm_aos,
               _temp_fock_prims, _temp_fock_aos,        dt=None):
    if dt == None:
        dt = tdscf.dt

    _p_prim_2,   _p_ao_2,\
    _f_prim_2,   _f_ao_2  = prop_step(
        tdscf, _temp_ts[2], _temp_dm_prims[2], _temp_fock_prims[2], dt = dt
        )

    _temp_dm_prims[4],   _temp_dm_aos[4],\
    _temp_fock_prims[4], _temp_fock_aos[4]  = prop_step(
        tdscf, _temp_ts[2], _temp_dm_prims[2], _f_prim_2, dt = dt
        )

    _temp_dm_prims[2]   = _temp_dm_prims[4]
    _temp_dm_aos[2]     = _temp_dm_aos[4]
    _temp_fock_prims[2] = _temp_fock_prims[4]
    _temp_fock_aos[2]   = _temp_fock_aos[4]

def mmut_prop(tdscf,                  
               _temp_ts,         _temp_dm_prims,   _temp_dm_aos,
               _temp_fock_prims, _temp_fock_aos,        dt=None):
#     # HF not numerically stable, for some reason
    if dt == None:
        dt = tdscf.dt

    _temp_dm_prims[3], _temp_dm_aos[3] = prop_step(
        tdscf, _temp_ts[1], _temp_dm_prims[1], _temp_fock_prims[2], dt = dt,
        build_fock=False
        )
    
    _temp_dm_prims[4],   _temp_dm_aos[4],\
    _temp_fock_prims[4], _temp_fock_aos[4] = prop_step(
        tdscf, _temp_ts[3], _temp_dm_prims[3], _temp_fock_prims[2], dt = dt/2
        )

    _temp_dm_prims[2]   = _temp_dm_prims[4]
    _temp_dm_aos[2]     = _temp_dm_aos[4]
    _temp_fock_prims[2] = _temp_fock_prims[4]
    _temp_fock_aos[2]   = _temp_fock_aos[4]

    _temp_dm_prims[1]   = _temp_dm_prims[3]
    _temp_dm_aos[1]     = _temp_dm_aos[3]

def lflp_pc_prop(tdscf, _temp_ts, _temp_dm_prims, _temp_fock_prims
                 , dt=None, tol = 1e-6):
    pass
#     if dt == None:
#         dt = tdscf.dt
#     temp5_t_ = dt + _temp_ts

#     step_converged = False
#     inner_iter = 0

#     _fock_prim_next_half_p = 2*_temp_fock_prims[2] - _temp_fock_prims[1]
#     if tdscf.verbose >= logger.DEBUG:
#         print_cx_matrix("_temp_fock_prims[2]", _temp_fock_prims[2])
#         print_cx_matrix("_temp_fock_prims[1]", _temp_fock_prims[1])
#     while (not step_converged) and inner_iter <= 200:
#         inner_iter += 1
#         _temp_dm_prims[4] = prop_step(tdscf, _temp_dm_prims[2], _fock_prim_next_half_p, dt)
#         _temp_dm_prims_next_half_c = (_temp_dm_prims[4] + _temp_dm_prims[2])/2
#         _fock_prim_next_half_c = tdscf.get_fock_prim(_temp_ts[3], _temp_dm_prims_next_half_c)
#         err = merr(_fock_prim_next_half_p, _fock_prim_next_half_c)
#         logger.debug(tdscf, "inner_iter = %d, err = %f", inner_iter, err)
#         # print(tdscf, "inner_iter = %d, err = %f"%(inner_iter, err))
#         _fock_prim_next_half_p = (_fock_prim_next_half_p + _fock_prim_next_half_c)/2
#         step_converged = (err<tol)
#         if tdscf.verbose >= logger.DEBUG:
#             print_cx_matrix("_fock_prim_next_half_p", _fock_prim_next_half_p)
#             print_cx_matrix("_fock_prim_next_half_c", _fock_prim_next_half_c )

#     if not step_converged:
#         logger.warn(tdscf, 'Inner loop not converged inner_iter = %d, err = %g', inner_iter, err)
#         raise RuntimeError("Inner loop not converged")
#     else:
#         _temp_dm_prims[3]   = _temp_dm_prims_next_half_c
#         _temp_fock_prims[4] = tdscf.get_fock_prim(_temp_ts[4], _temp_dm_prims[4])

#     temp_fock_prim_ = np.zeros(_temp_dm_prims.shape, dtype=np.complex128)
#     temp_temp_dm_prims_   = np.zeros(_temp_dm_prims.shape, dtype=np.complex128)
    
#     temp_temp_dm_prims_[0] = _temp_dm_prims[2]
#     temp_temp_dm_prims_[1] = _temp_dm_prims[3]
#     temp_temp_dm_prims_[2] = _temp_dm_prims[4]

#     temp_fock_prim_[0] = _temp_fock_prims[2]
#     temp_fock_prim_[1] = (_fock_prim_next_half_c + _fock_prim_next_half_p)/2
#     temp_fock_prim_[2] = _temp_fock_prims[4]

#     return temp5_t_, temp_temp_dm_prims_, temp_fock_prim_

def ep_pc_prop(tdscf, _temp_ts, _temp_dm_prims, _temp_fock_prims
                 , dt=None, tol = 1e-6):
    pass


def orth_ao(tdscf, key="canonical"):
    s1e = tdscf.mf.get_ovlp().astype(np.complex128)
    if key.lower() == "canonical":
        logger.info(tdscf, "the AOs are orthogonalized with canonical MO coefficients")
        if not tdscf.mf.converged:
            raise RuntimeError("the RT TDSCF object must be initialzed with a converged SCF object")

        x = tdscf.mf.mo_coeff.astype(np.complex128)
        x_t = x.T
        x_inv = np.einsum('li,ls->is', x, s1e)
        x_t_inv = x_inv.T
        tdscf.orth_xtuple = (x, x_t, x_inv, x_t_inv)
        return tdscf.orth_xtuple
    else:
        x = lo.orth_ao(tdscf.mol, method=key).astype(np.complex128)
        x_t = x.T
        x_inv = np.einsum('li,ls->is', x, s1e)
        x_t_inv = x_inv.T
        tdscf.orth_xtuple = (x, x_t, x_inv, x_t_inv)
        return tdscf.orth_xtuple

def ao2orth_dm(dm_ao, orth_xtuple):
    x, x_t, x_inv, x_t_inv = orth_xtuple
    dm_prim = reduce(np.dot, (x_inv, dm_ao, x_t_inv))
    return dm_prim

def orth2ao_dm(dm_prim, orth_xtuple):
    x, x_t, x_inv, x_t_inv = orth_xtuple
    dm_ao = reduce(np.dot, (x, dm_prim, x_t))
    return dm_ao# (dm_ao + dm_ao.conj().T)/2

def ao2orth_fock(fock_ao, orth_xtuple):
    x, x_t, x_inv, x_t_inv = orth_xtuple
    fock_prim = reduce(np.dot, (x_t, fock_ao, x))
    return fock_prim

def orth2ao_fock(fock_prim, orth_xtuple):
    x, x_t, x_inv, x_t_inv = orth_xtuple
    fock_ao = reduce(np.dot, (x_t_inv, fock_prim, x_inv))
    return fock_ao # (fock_ao + fock_ao.conj().T)/2

def kernel(tdscf,                                #input
           dt        = None, maxstep     = None, #input
           dm_ao_init= None, prop_func   = None, #input
           ndm_prim  = None, nfock_prim  = None, #output
           ndm_ao    = None, nfock_ao    = None, #output
           netot     = None, do_dump_chk = True
           ):
    cput0 = (time.clock(), time.time())

    if dt == None:          dt = tdscf.dt
    if maxstep == None:     maxstep = tdscf.maxstep
    if dm_ao_init is None:  dm_ao_init = tdscf.dm_ao_init
    if prop_func == None:   prop_func = tdscf.prop_func

    if ndm_prim is None:
        ndm_prim = tdscf.ndm_prim
    if nfock_prim is None:
        nfock_prim = tdscf.nfock_prim
    
    h1e_ao = tdscf.mf.get_hcore()
    if tdscf.efield_vec is None:
        tdscf.mf.get_hcore = lambda *args, t=0.0: h1e_ao
    else:
        tdscf.mf.get_hcore = lambda *args, t=0.0: (
            h1e_ao + np.einsum('xij,x->ij', tdscf.ele_dip_ao, tdscf.efield_vec(t) )
            )

    if tdscf.verbose >= logger.DEBUG1:
            print_matrix("The field-free hcore matrix  is, ", h1e_ao, ncols=PRINT_MAT_NCOL)
    if tdscf.verbose >= logger.DEBUG1:
            print_matrix("The t=1.0 a.u. efield matrix is, ", tdscf.mf.get_hcore(t=1.0), ncols=PRINT_MAT_NCOL)

    dm_ao_init   = dm_ao_init.astype(np.complex128)
    dm_prim_init = ao2orth_dm(dm_ao_init, tdscf.orth_xtuple)

    fock_ao_init = (tdscf.mf.get_fock(dm=dm_ao_init, h1e=tdscf.mf.get_hcore(t=0.0)))
    fock_prim_init = ao2orth_fock(fock_ao_init, tdscf.orth_xtuple)

    etot_init      = tdscf.mf.energy_tot(dm=dm_ao_init, h1e=tdscf.mf.get_hcore(t=0.0))

    if tdscf.verbose >= logger.DEBUG1:
        print_matrix("the initial dm prim is", dm_prim_init, ncols=PRINT_MAT_NCOL)
        print_matrix("the initial fock prim without electric field is", 
                     ao2orth_fock(
                         tdscf.mf.get_fock(dm=dm_ao_init).real, tdscf.orth_xtuple
                         ), ncols=PRINT_MAT_NCOL
                     )
        print_matrix("the initial fock prim with electric field is"
                     , fock_prim_init, ncols=PRINT_MAT_NCOL)

    shape = list(dm_ao_init.shape)

    ndm_prim[0]    = dm_prim_init
    nfock_prim[0]  = fock_prim_init
    ndm_ao[0]      = dm_ao_init
    nfock_ao[0]    = fock_ao_init
    netot[0]       = etot_init

    _temp_ts         = dt*np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    _temp_dm_prims   = np.zeros([5] + shape, dtype=np.complex128)
    _temp_fock_prims = np.zeros([5] + shape, dtype=np.complex128)
    _temp_dm_aos     = np.zeros([5] + shape, dtype=np.complex128)
    _temp_fock_aos   = np.zeros([5] + shape, dtype=np.complex128)

    cput1 = logger.timer(tdscf, 'initialize td-scf', *cput0)

# propagation start here
    _temp_dm_prims[0]   = ndm_prim[0]
    _temp_fock_prims[0] = nfock_prim[0]
    _temp_dm_aos[0]     = ndm_ao[0]
    _temp_fock_aos[0]   = nfock_ao[0]

    _temp_dm_prims[1],   _temp_dm_aos[1],\
    _temp_fock_prims[1], _temp_fock_aos[1] = prop_step(
        tdscf, _temp_ts[0], _temp_dm_prims[0], _temp_fock_prims[0], dt = dt/2
        )
    _temp_dm_prims[2],   _temp_dm_aos[2],\
    _temp_fock_prims[2], _temp_fock_aos[2] = prop_step(
        tdscf, _temp_ts[1], _temp_dm_prims[1], _temp_fock_prims[1], dt = dt/2
        )

    istep = 1
    while istep <= maxstep:
        if istep%100==1:
            logger.note(tdscf, 'istep=%d, time=%f, delta e=%e', istep-1, tdscf.ntime[istep-1], tdscf.netot[istep-1]-tdscf.netot[0])
        # propagation step
        prop_func(tdscf, _temp_ts, _temp_dm_prims, _temp_dm_aos, 
                                           _temp_fock_prims, _temp_fock_aos, dt=dt)
        ndm_prim[istep]   =   _temp_dm_prims[2]
        ndm_ao[istep]     =     _temp_dm_aos[2]
        nfock_prim[istep] = _temp_fock_prims[2]
        nfock_ao[istep]   =   _temp_fock_aos[2]
        netot[istep]      =   tdscf.mf.energy_tot(
            dm=_temp_dm_aos[2], h1e=tdscf.mf.get_hcore(t=_temp_ts[2])
        )
        _temp_ts = _temp_ts + dt
        istep += 1
    cput2 = logger.timer(tdscf, 'propagation %d time steps'%(istep-1), *cput0)

    if do_dump_chk and tdscf.chkfile:
        ntime = tdscf.ntime
        nstep = tdscf.nstep
        tdscf.dump_chk(locals())
        cput3 = logger.timer(tdscf, 'dump chk finished', *cput0)
   

class TDSCF(lib.StreamObject):
    def __init__(self, mf):
# the class that defines the system, mol and mf
        if not mf.converged:
            logger.warn(self, "SCF not converged, RT-TDSCF method should be initialized with a converged SCF")
        self.mf             = mf
        self.mol            = mf.mol
        self.verbose        = mf.verbose
        self.mf.verbose     = 0
        self.max_memory     = mf.max_memory
        self.stdout         = mf.stdout
        self.orth_xtuple    = None
# the interaction between the system and electric field
        self.ele_dip_ao   = self.mf.mol.intor_symmetric('int1e_r', comp=3)

# If chkfile is muted, SCF intermediates will not be dumped anywhere.
        if MUTE_CHKFILE:
            self.chkfile = None
        else:
# the chkfile will be removed automatically, to save the chkfile, assign a
# filename to self.chkfile
            self._chkfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
            self.chkfile = self._chkfile.name

# input parameters for propagation
# initial condtion
        self.dm_ao_init = None

# time step and maxstep
        self.dt         = None
        self.maxstep    = None

# propagation method
        self.prop_method  = None # a string
        self.orth_method  = None # a string
        self.prop_func    = None # may not define directly here

# electric field during propagation, a function that returns a
# 3-component vector.
        self.efield_vec  = None

# don't modify the following attributes, they are not input options
        self.nstep       = None
        self.ntime       = None
        self.ndm_prim    = None
        self.nfock_prim  = None
        self.ndm_ao      = None
        self.nfock_ao    = None
        self.netot       = None

    def set_prop_func(self, key='mmut'):
        '''
        In virtually all cases AMUT is superior in terms of stability. 
        Others are perhaps only useful for debugging or simplicity.
        '''
        if key.lower() == 'amut1' or key.lower() == 'amut':
            self.prop_func = amut1_prop
        elif key.lower() == 'amut2':
            self.prop_func = amut2_prop
        elif key.lower() == 'amut3':
            self.prop_func = amut3_prop
        elif key.lower() == 'aeut':
            self.prop_func = aeut_prop
        elif key.lower() == 'euler':
            self.prop_func = euler_prop
        elif key.lower() == 'lflp_pc':
            self.prop_func = lflp_pc_prop
        elif key.lower() == 'mmut':
            self.prop_func = mmut_prop
        else:
            raise RuntimeError("unknown prop method!")

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        if log.verbose < logger.INFO:
            return self
        log.info('\n')
        log.info('******** %s ********', self.__class__)
        log.info(
        'This is a real time TDSCF calculation initialized with a %s SCF',
            (
            "converged" if self.mf.converged else "not converged"
            )
        )
        if self.mf.converged:
            log.info(
                'The SCF converged tolerence is conv_tol = %g, conv_tol should be less that 1e-8'%self.mf.conv_tol
                )
        if hasattr(self.mf, 'xc'):
            log.info(
            'The initial condition is a RKS instance, the xc functional is %s'%self.mf.xc
            )
        else:
            log.info(
            'The initial condition is a HF instance'
            )
        if self.chkfile:
            log.info('chkfile to save RT TDSCF result = %s', self.chkfile)
        log.info( 'dt = %f, maxstep = %d', self.dt, self.maxstep )
        log.info( 'prop_method = %s', self.prop_func.__name__)
        log.info('max_memory %d MB (current use %d MB)', self.max_memory, lib.current_memory()[0])
        return self

    def _initialize(self):
        if self.prop_func is None:
            if self.prop_method is not None:
                self.set_prop_func(key=self.prop_method)
            else:
                self.set_prop_func()
        self.dump_flags()

        if self.orth_method is None:
            self.orth_xtuple = orth_ao(self)
        else:
            logger.info(self, 'orth method is %s.', self.orth_method)
            self.orth_xtuple = orth_ao(self, key=self.orth_method)
        
        if self.verbose >= logger.DEBUG1:
            print_matrix(
                "XT S X", reduce(np.dot, (self.orth_xtuple[1], self.mf.get_ovlp(), self.orth_xtuple[0]))
                , ncols=PRINT_MAT_NCOL)

        if self.verbose >= logger.DEBUG1:
                print_matrix("The field-free hcore matrix  is, ", self.mf.get_hcore(), ncols=PRINT_MAT_NCOL)
        if self.verbose >= logger.DEBUG1:
                print_matrix("The t=1.0 a.u. efield matrix is, ", self.get_efield(1.0), ncols=PRINT_MAT_NCOL)

    def _finalize(self):
        self.ndipole = np.zeros([self.maxstep+1,             3])
        self.npop    = np.zeros([self.maxstep+1, self.mol.natm])
        logger.info(self, "Finalization begins here")
        s1e = self.mf.get_ovlp()
        for i,idm in enumerate(self.ndm_ao):
            self.ndipole[i] = self.mf.dip_moment(dm = idm.real, unit='au', verbose=0)
            self.npop[i]    = self.mf.mulliken_pop(dm = idm.real, s=s1e, verbose=0)[1]
        logger.info(self, "Finalization finished")

    def kernel(self, dm_ao_init=None):
        self._initialize()
        if dm_ao_init is None:
            if self.dm_ao_init is not None:
                dm_ao_init = self.dm_ao_init
            elif self.dm_ao_init == None:
                dm_ao_init = self.mf.make_rdm1()
        logger.info(self, "Propagation begins here")
        if self.verbose >= logger.DEBUG1:
                print_matrix("The initial density matrix is, ", dm_ao_init, ncols=PRINT_MAT_NCOL)

        logger.info(self, 'before building matrices, max_memory %d MB (current use %d MB)', self.max_memory, lib.current_memory()[0])
        self.nstep      = np.linspace(0, self.maxstep, self.maxstep+1, dtype=int) # output
        self.ntime      = self.dt*self.nstep                                      # output
        self.ndm_prim   = np.zeros([self.maxstep+1] + list(dm_ao_init.shape), dtype=np.complex128) # output
        self.ndm_ao     = np.zeros([self.maxstep+1] + list(dm_ao_init.shape), dtype=np.complex128) # output
        self.nfock_prim = np.zeros([self.maxstep+1] + list(dm_ao_init.shape), dtype=np.complex128) # output
        self.nfock_ao   = np.zeros([self.maxstep+1] + list(dm_ao_init.shape), dtype=np.complex128) # output
        self.netot      = np.zeros([self.maxstep+1])                                # output
        logger.info(self, 'after building matrices, max_memory %d MB (current use %d MB)', self.max_memory, lib.current_memory()[0])
        kernel(
           self,                                                        #input
           dt        = self.dt         , maxstep     = self.maxstep,    #input
           dm_ao_init= dm_ao_init      , prop_func   = self.prop_func,  #input
           ndm_prim  = self.ndm_prim   , nfock_prim  = self.nfock_prim, #output
           ndm_ao    = self.ndm_ao     , nfock_ao    = self.nfock_ao,   #output
           netot     = self.netot
            )
        logger.info(self, 'after propogation matrices, max_memory %d MB (current use %d MB)', self.max_memory, lib.current_memory()[0])
        logger.info(self, "Propagation finished")
        self._finalize()

    def dump_chk(self, envs):
        if self.chkfile:
            logger.info(self, 'chkfile to save RT TDSCF result is %s', self.chkfile)
            chkfile.dump_rt(self.mol, self.chkfile,
                             envs['nstep'], envs['ntime'],
                             envs['ndm_prim'], envs['ndm_ao'],
                             envs['nfock_prim'], envs['nfock_ao'],
                             envs['netot'],
                             overwrite_mol=False)

