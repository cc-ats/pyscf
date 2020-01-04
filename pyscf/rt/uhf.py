import time
import tempfile

from functools import reduce

import numpy
import scipy

from pyscf import gto, scf
from pyscf import lib, lo
from pyscf.lib import logger
from pyscf.rt import chkfile

from pyscf.rt import rhf as rhf_tdscf
from pyscf.rt.util import build_absorption_spectrum
from pyscf.rt.util import print_matrix, print_cx_matrix, errm, expm

from pyscf import __config__

REF_BASIS         = getattr(__config__, 'lo_orth_pre_orth_ao_method', 'ANO'      )
MUTE_CHKFILE      = getattr(__config__, 'rt_tdscf_mute_chkfile',      False      )
PRINT_MAT_NCOL    = getattr(__config__, 'rt_tdscf_print_mat_ncol',    7          )
ORTH_METHOD       = getattr(__config__, 'rt_tdscf_orth_ao_method',    'canonical')

# re-define Orthogonalize AOs for UHF
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
    from pyscf.lo import nao
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
        c_orth_a = numpy.dot(pre_orth_ao, lowdin(s1))
        c_orth_b = numpy.dot(pre_orth_ao, lowdin(s1))

    elif method.lower() == 'nao':
        from pyscf.lo import nao
        assert(mf is not None)
        logger.info(mf, "the AOs are orthogonalized with NAO")
        c_orth_a = nao.nao(mol, mf, s)
        c_orth_b = nao.nao(mol, mf, s)

    elif method.lower() == 'canonical':
        assert(mf is not None)
        logger.info(mf, "the AOs are orthogonalized with canonical MO coefficients")
        if not mf.converged:
            raise RuntimeError("the MF must be converged")
        c_orth_a = mf.mo_coeff[0]
        c_orth_b = mf.mo_coeff[1]

    else: # meta_lowdin: divide ao into core, valence and Rydberg sets,
          # orthogonalizing within each set
        weight = numpy.ones(pre_orth_ao.shape[0])
        c_orth_a = nao._nao_sub(mol, weight, pre_orth_ao, s)
        c_orth_b = nao._nao_sub(mol, weight, pre_orth_ao, s)
    # adjust phase
    for i in range(c_orth_a.shape[1]):
        if c_orth_a[i,i] < 0:
            c_orth_a[:,i] *= -1
        if c_orth_b[i,i] < 0:
            c_orth_b[:,i] *= -1
    return numpy.array((c_orth_a, c_orth_b)).astype(numpy.complex128)

def ao2orth_dm(tdscf, dm_ao):
    x, x_t, x_inv, x_t_inv = tdscf.orth_xtuple
    dm_prim_a = reduce(numpy.dot, (x_inv[0], dm_ao[0], x_t_inv[0]))
    dm_prim_b = reduce(numpy.dot, (x_inv[1], dm_ao[1], x_t_inv[1]))
    return numpy.array((dm_prim_a, dm_prim_b))

def orth2ao_dm(tdscf, dm_prim):
    x, x_t, x_inv, x_t_inv = tdscf.orth_xtuple
    dm_ao_a = reduce(numpy.dot, (x[0], dm_prim[0], x_t[0]))
    dm_ao_b = reduce(numpy.dot, (x[1], dm_prim[1], x_t[1]))
    return numpy.array((dm_ao_a, dm_ao_b))# (dm_ao + dm_ao.conj().T)/2

def ao2orth_fock(tdscf, fock_ao):
    x, x_t, x_inv, x_t_inv = tdscf.orth_xtuple
    fock_prim_a = reduce(numpy.dot, (x_t[0], fock_ao[0], x[0]))
    fock_prim_b = reduce(numpy.dot, (x_t[1], fock_ao[1], x[1]))
    return numpy.array((fock_prim_a, fock_prim_b))

def orth2ao_fock(tdscf, fock_prim):
    x, x_t, x_inv, x_t_inv = tdscf.orth_xtuple
    fock_ao_a = reduce(numpy.dot, (x_t_inv[0], fock_prim[0], x_inv[0]))
    fock_ao_b = reduce(numpy.dot, (x_t_inv[1], fock_prim[1], x_inv[1]))
    return numpy.array((fock_ao_a, fock_ao_b)) # (fock_ao + fock_ao.conj().T)/2

# propagate step
def prop_step(tdscf, dt, fock_prim, dm_prim):
    propogator_a = expm(-1j*dt*fock_prim[0])
    propogator_b = expm(-1j*dt*fock_prim[1])

    dm_prim_a_   = reduce(numpy.dot, [propogator_a, dm_prim[0], propogator_a.conj().T])
    dm_prim_a_   = (dm_prim_a_ + dm_prim_a_.conj().T)/2
    dm_prim_b_   = reduce(numpy.dot, [propogator_b, dm_prim[1], propogator_b.conj().T])
    dm_prim_b_   = (dm_prim_b_ + dm_prim_b_.conj().T)/2

    dm_prim_     = numpy.array((dm_prim_a_, dm_prim_b_))
    dm_ao_       = tdscf.orth2ao_dm(  dm_prim_)
    
    return dm_prim_, dm_ao_

LAST      = 0 
LAST_HALF = 1
THIS      = 2
NEXT_HALF = 3
NEXT      = 4

   
class TDHF(rhf_tdscf.TDHF):
    prop_step = prop_step
    ao2orth_dm = ao2orth_dm
    orth2ao_dm = orth2ao_dm
    ao2orth_fock = ao2orth_fock
    orth2ao_fock = orth2ao_fock

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
            'The initial condition is a UHF instance'
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
            x = orth_ao(self.mf)
            x_t = numpy.einsum('aij->aji', x)
            x_inv = numpy.einsum('ali,ls->ais', x, self.mf.get_ovlp())
            x_t_inv = numpy.einsum('aij->aji', x_inv)
            self.orth_xtuple = (x, x_t, x_inv, x_t_inv)
        else:
            logger.info(self, 'orth method is %s.', self.orth_method)
            x = orth_ao(self.mf, method=self.orth_method)
            x_t = numpy.einsum('aij->aji', x)
            x_inv = numpy.einsum('ali,ls->ais', x, self.mf.get_ovlp())
            x_t_inv = numpy.einsum('aij->aji', x_inv)
            self.orth_xtuple = (x, x_t, x_inv, x_t_inv)
        
        if self.verbose >= logger.DEBUG1:
            print_matrix(
                "alpha XT S X", reduce(numpy.dot, (self.orth_xtuple[1][0], self.mf.get_ovlp(), self.orth_xtuple[0][0]))
                , ncols=PRINT_MAT_NCOL)
            print_matrix(
                "beta  XT S X", reduce(numpy.dot, (self.orth_xtuple[1][1], self.mf.get_ovlp(), self.orth_xtuple[0][1]))
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
    O    0.0000000    0.0000000    0.5754646
    O    0.0000000    0.0000000   -0.5754646
    '''
    , basis='cc-pvdz', spin=2, symmetry=False).build()

    mf = scf.UHF(mol)
    mf.verbose = 5
    mf.kernel()

    dm = mf.make_rdm1()
    fock = mf.get_fock()

    rttd = TDHF(mf)
    rttd.verbose = 5
    rttd.maxstep = 100
    rttd.dt      = 0.1
    rttd.kernel(dm_ao_init=dm)
    print(rttd.netot)

    print_matrix("canonical dm", rttd.ao2orth_dm(dm) )

    
