import time
import tempfile

from functools import reduce

import numpy
import scipy

from pyscf import gto, scf
from pyscf import lib, lo
from pyscf.lib import logger
from pyscf.rt  import chkfile

from pyscf.rt.util import build_absorption_spectrum
from pyscf.rt.util import print_matrix, print_cx_matrix, errm, expm

from pyscf import __config__

REF_BASIS         = getattr(__config__, 'lo_orth_pre_orth_ao_method', 'ANO'      )
MUTE_CHKFILE      = getattr(__config__, 'rt_tdscf_mute_chkfile',      False      )
PRINT_MAT_NCOL    = getattr(__config__, 'rt_tdscf_print_mat_ncol',    7          )
ORTH_METHOD       = getattr(__config__, 'rt_tdscf_orth_ao_method',    'canonical')

# re-define Orthogonalize AOs
def orth_ao(mf_or_mol, method=ORTH_METHOD, pre_orth_ao=None, scf_method=None,
            s=None):
    '''Orthogonalize AOs
    Kwargs:
        method : str
            One of
            | lowdin : Symmetric orthogonalization
            | meta-lowdin : Lowdin orth within core, valence, virtual space separately (JCTC, 10, 3784)
            | canonical MO
            | NAO
    '''

    mf = scf_method
    if isinstance(mf_or_mol, gto.Mole):
        mol = mf_or_mol
    else:
        mol = mf_or_mol.mol
        if mf is None:
            mf = mf_or_mol

    if s is None:
        if hasattr(mol, 'pbc_intor'):  # whether mol object is a cell
            s = mol.pbc_intor('int1e_ovlp', hermi=1)
        else:
            s = mol.intor_symmetric('int1e_ovlp')

    if pre_orth_ao is None:
#        pre_orth_ao = numpy.eye(mol.nao_nr())
        from pyscf.lo.orth import project_to_atomic_orbitals
        pre_orth_ao = project_to_atomic_orbitals(mol, REF_BASIS)

    if method.lower() == 'lowdin':
        from pyscf.lo import lowdin
        logger.info(mf, "the AOs are orthogonalized with Lowdin")
        s1 = reduce(numpy.dot, (pre_orth_ao.conj().T, s, pre_orth_ao))
        c_orth = numpy.dot(pre_orth_ao, lowdin(s1))

    elif method.lower() == 'nao':
        from pyscf.lo import nao
        assert(mf is not None)
        logger.info(mf, "the AOs are orthogonalized with NAO")
        c_orth = nao.nao(mol, mf, s)

    elif method.lower() == 'canonical':
        assert(mf is not None)
        logger.info(mf, "the AOs are orthogonalized with canonical MO coefficients")
        if not mf.converged:
            raise RuntimeError("the MF must be converged")
        c_orth = mf.mo_coeff

    else: # meta_lowdin: divide ao into core, valence and Rydberg sets,
          # orthogonalizing within each set
        weight = numpy.ones(pre_orth_ao.shape[0])
        c_orth = nao._nao_sub(mol, weight, pre_orth_ao, s)
    # adjust phase
    for i in range(c_orth.shape[1]):
        if c_orth[i,i] < 0:
            c_orth[:,i] *= -1
    return c_orth.astype(numpy.complex128)

def ao2orth_dm(dm_ao, orth_xtuple):
    x, x_t, x_inv, x_t_inv = orth_xtuple
    dm_prim = reduce(numpy.dot, (x_inv, dm_ao, x_t_inv))
    return dm_prim

def orth2ao_dm(dm_prim, orth_xtuple):
    x, x_t, x_inv, x_t_inv = orth_xtuple
    dm_ao = reduce(numpy.dot, (x, dm_prim, x_t))
    return dm_ao# (dm_ao + dm_ao.conj().T)/2

def ao2orth_fock(fock_ao, orth_xtuple):
    x, x_t, x_inv, x_t_inv = orth_xtuple
    fock_prim = reduce(numpy.dot, (x_t, fock_ao, x))
    return fock_prim

def orth2ao_fock(fock_prim, orth_xtuple):
    x, x_t, x_inv, x_t_inv = orth_xtuple
    fock_ao = reduce(numpy.dot, (x_t_inv, fock_prim, x_inv))
    return fock_ao # (fock_ao + fock_ao.conj().T)/2

# propagate step
def prop_step(tdscf, t_start, t_end, fock_prim, dm_prim, 
              build_fock=True, h1e=None):
    dt = t_end - t_start
    assert dt > 0 # may be removed 
    propogator = expm(-1j*dt*fock_prim)
    dm_prim_   = reduce(numpy.dot, [propogator, dm_prim, propogator.conj().T])
    dm_prim_   = (dm_prim_ + dm_prim_.conj().T)/2
    dm_ao_     = orth2ao_dm(dm_prim_, tdscf.orth_xtuple)
    
    if build_fock and (h1e is not None):
        fock_ao_   = tdscf.mf.get_fock(
            dm=dm_ao_, h1e=h1e
        )
        fock_prim_ = ao2orth_fock(fock_ao_, tdscf.orth_xtuple)
        return dm_prim_, dm_ao_, fock_prim_, fock_ao_
    else:
        return dm_prim_, dm_ao_

LAST      = 0 
LAST_HALF = 1
THIS      = 2
NEXT_HALF = 3
NEXT      = 4

def euler_prop(tdscf,  _temp_ts,
               _temp_dm_prims,   _temp_dm_aos,
               _temp_fock_prims, _temp_fock_aos):
    _temp_dm_prims[NEXT],   _temp_dm_aos[NEXT],\
    _temp_fock_prims[NEXT], _temp_fock_aos[NEXT] = prop_step(
        tdscf, _temp_ts[THIS],  _temp_ts[NEXT],
        _temp_fock_prims[THIS], _temp_dm_prims[THIS],
        build_fock = True,      h1e = tdscf.get_hcore(_temp_ts[NEXT])
        )
    _temp_dm_prims[THIS]   = _temp_dm_prims[NEXT]
    _temp_dm_aos[THIS]     = _temp_dm_aos[NEXT]
    _temp_fock_prims[THIS] = _temp_fock_prims[NEXT]
    _temp_fock_aos[THIS]   = _temp_fock_aos[NEXT]

def kernel(tdscf,                                #input
           dt        = None, maxstep     = None, #input
           dm_ao_init= None, prop_func   = None, #input
           ndm_prim  = None, nfock_prim  = None, #output
           ndm_ao    = None, nfock_ao    = None, #output
           netot     = None, do_dump_chk = True
           ):
    cput0 = (time.clock(), time.time())

    if dt is None:          dt = tdscf.dt
    if maxstep == None:     maxstep = tdscf.maxstep
    if dm_ao_init is None:  dm_ao_init = tdscf.dm_ao_init
    if prop_func == None:   prop_func = tdscf.prop_func

    if ndm_prim is None:
        ndm_prim = tdscf.ndm_prim
    if nfock_prim is None:
        nfock_prim = tdscf.nfock_prim

    dm_ao_init   = dm_ao_init.astype(numpy.complex128)
    dm_prim_init = ao2orth_dm(dm_ao_init, tdscf.orth_xtuple)

    fock_ao_init   = (tdscf.mf.get_fock(dm=dm_ao_init, h1e=tdscf.get_hcore(t=0.0)))
    fock_prim_init = ao2orth_fock(fock_ao_init, tdscf.orth_xtuple)

    etot_init      = tdscf.mf.energy_tot(dm=dm_ao_init, h1e=tdscf.get_hcore(t=0.0)).real

    shape = list(dm_ao_init.shape)

    ndm_prim[0]    = dm_prim_init
    nfock_prim[0]  = fock_prim_init
    ndm_ao[0]      = dm_ao_init
    nfock_ao[0]    = fock_ao_init
    netot[0]       = etot_init

    _temp_ts         = dt*numpy.array([0.0, 0.5, 1.0, 1.5, 2.0])
    _temp_dm_prims   = numpy.zeros([5] + shape, dtype=numpy.complex128)
    _temp_fock_prims = numpy.zeros([5] + shape, dtype=numpy.complex128)
    _temp_dm_aos     = numpy.zeros([5] + shape, dtype=numpy.complex128)
    _temp_fock_aos   = numpy.zeros([5] + shape, dtype=numpy.complex128)

    cput1 = logger.timer(tdscf, 'initialize td-scf', *cput0)

# propagation start here
    _temp_dm_prims[LAST]   = ndm_prim[0]
    _temp_fock_prims[LAST] = nfock_prim[0]
    _temp_dm_aos[LAST]     = ndm_ao[0]
    _temp_fock_aos[LAST]   = nfock_ao[0]

    _temp_dm_prims[LAST_HALF],   _temp_dm_aos[LAST_HALF],\
    _temp_fock_prims[LAST_HALF], _temp_fock_aos[LAST_HALF] = prop_step(
        tdscf, _temp_ts[LAST],  _temp_ts[LAST_HALF],
        _temp_fock_prims[LAST], _temp_dm_prims[LAST],
        build_fock = True,      h1e = tdscf.get_hcore(_temp_ts[LAST_HALF])
        )
    _temp_dm_prims[THIS],   _temp_dm_aos[THIS],\
    _temp_fock_prims[THIS], _temp_fock_aos[THIS] = prop_step(
        tdscf, _temp_ts[LAST_HALF],  _temp_ts[THIS],
        _temp_fock_prims[LAST_HALF], _temp_dm_prims[LAST_HALF],
        build_fock = True,      h1e = tdscf.get_hcore(_temp_ts[THIS])
        )

    istep = 1
    while istep <= maxstep:
        if istep%100==1:
            logger.note(tdscf, 'istep=%d, time=%f, delta e=%e',
            istep-1, tdscf.ntime[istep-1], tdscf.netot[istep-1]-tdscf.netot[0])
        # propagation step
        prop_func(tdscf,              _temp_ts,
               _temp_dm_prims,    _temp_dm_aos,
               _temp_fock_prims, _temp_fock_aos
               )
        ndm_prim[istep]   =   _temp_dm_prims[THIS]
        ndm_ao[istep]     =     _temp_dm_aos[THIS]
        nfock_prim[istep] = _temp_fock_prims[THIS]
        nfock_ao[istep]   =   _temp_fock_aos[THIS]
        netot[istep]      =   tdscf.mf.energy_tot(
            dm=_temp_dm_aos[THIS], h1e=tdscf.get_hcore(_temp_ts[THIS])
        ).real
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

# electric field during propagation, a function that returns a
# 3-component vector.
        self.efield_vec  = None
# the interaction between the system and electric field
        self.ele_dip_ao  = None

# mf information
        self.ovlp_ao         = self.mf.get_ovlp().astype(numpy.complex128)
        self.hcore_ao        = self.mf.get_hcore().astype(numpy.complex128)

# propagation method
        self.prop_method  = None # a string
        self.orth_method  = None # a string
        self.prop_func    = None # may not define directly here

# don't modify the following attributes, they are not input options
        self.nstep       = None
        self.ntime       = None
        self.ndm_prim    = None
        self.nfock_prim  = None
        self.ndm_ao      = None
        self.nfock_ao    = None
        self.netot       = None

    def get_hcore(self, t=None):
        if (self.efield_vec is None) or (t is None):
            return self.hcore_ao
        else:
            if self.ele_dip_ao is None:
                # the interaction between the system and electric field
                self.ele_dip_ao      = self.mf.mol.intor_symmetric('int1e_r', comp=3)
            h = self.hcore_ao + numpy.einsum(
                'xij,x->ij', self.ele_dip_ao, self.efield_vec(t)
                ).astype(numpy.complex128)
            return h

    def set_prop_func(self, key='euler'):
        '''
        In virtually all cases AMUT is superior in terms of stability. 
        Others are perhaps only useful for debugging or simplicity.
        '''
        if (key is not None):
            if (key.lower() == 'euler'):
                self.prop_func = euler_prop
            else:
                raise RuntimeError("unknown prop method!")
        else:
            self.prop_func = euler_prop

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
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
        log.info(
            'The initial condition is a RHF instance'
            )
        if self.chkfile:
            log.info('chkfile to save RT TDSCF result = %s', self.chkfile)
        log.info( 'dt = %f, maxstep = %d', self.dt, self.maxstep )
        log.info( 'prop_method = %s', self.prop_func.__name__)
        log.info('max_memory %d MB (current use %d MB)', self.max_memory, lib.current_memory()[0])

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
            x = orth_ao(self.mf, method=self.orth_method)
            x_t = x.T
            x_inv = numpy.einsum('li,ls->is', x, self.mf.get_ovlp() )
            x_t_inv = x_inv.T

            self.orth_xtuple = (x, x_t, x_inv, x_t_inv)
        
        if self.verbose >= logger.DEBUG1:
            print_matrix(
                "XT S X", reduce(numpy.dot, (self.orth_xtuple[1], self.mf.get_ovlp(), self.orth_xtuple[0]))
                , ncols=PRINT_MAT_NCOL)

    def _finalize(self):
        self.ndipole = numpy.zeros([self.maxstep+1,             3])
        self.npop    = numpy.zeros([self.maxstep+1, self.mol.natm])
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
        self.nstep      = numpy.linspace(0, self.maxstep, self.maxstep+1, dtype=int) # output
        self.ntime      = self.dt*self.nstep                                      # output
        self.ndm_prim   = numpy.zeros([self.maxstep+1] + list(dm_ao_init.shape), dtype=numpy.complex128) # output
        self.ndm_ao     = numpy.zeros([self.maxstep+1] + list(dm_ao_init.shape), dtype=numpy.complex128) # output
        self.nfock_prim = numpy.zeros([self.maxstep+1] + list(dm_ao_init.shape), dtype=numpy.complex128) # output
        self.nfock_ao   = numpy.zeros([self.maxstep+1] + list(dm_ao_init.shape), dtype=numpy.complex128) # output
        self.netot      = numpy.zeros([self.maxstep+1])                                # output
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

if __name__ == "__main__":
    mol =   gto.Mole( atom='''
  H    0.0000000    0.0000000    0.3540000
  H    0.0000000    0.0000000   -0.3540000
    '''
    , basis='sto-3g', symmetry=False).build()

    mf = scf.RHF(mol)
    mf.verbose = 5
    mf.kernel()

    dm = mf.make_rdm1()
    fock = mf.get_fock()
    rttd = TDSCF(mf)
