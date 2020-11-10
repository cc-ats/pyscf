import time
import numpy
import copy
import scipy.linalg as la

import pyscf
from pyscf     import lib
from pyscf.lib import logger

from pyscf import gto
from pyscf import scf
from pyscf import dft
from pyscf.scf import cphf
from pyscf.dft import numint
from .frag_pop import FragmentPopulation

from copy import deepcopy
from functools import reduce

# import _response_functions to load gen_response methods in SCF class
from pyscf.scf import _response_functions  # noqa
from pyscf import __config__

def get_newton_step_aug_hess(grad, hess):
    #lamb = 1.0 / alpha
    ah = numpy.zeros((hess.shape[0]+1,hess.shape[1]+1))
    ah[1:,0]  = grad
    ah[0,1:]  = grad.conj()
    ah[1:,1:] = hess

    eigval, eigvec = la.eigh(ah)
    idx = None
    for i in range(len(eigvec)):
        if abs(eigvec[0,i]) > 0.1 and eigval[i] > 0.0:
            idx = i
            break
    if idx is None:
        print("WARNING: ALL EIGENVALUESS in AUG-HESSIAN are NEGATIVE!!! ")
        return numpy.zeros_like(grad)
    deltax = eigvec[1:,idx] / eigvec[0,idx]
    return deltax

class Constraints(object):
    def __init__(self, frag_pop, nelec_required_list):
        self._scf                 = frag_pop._scf
        self._mol                 = frag_pop._mol
        self._frag_pop            = frag_pop
        self._nelec_required_list = numpy.asarray(nelec_required_list, dtype=float)

        self.constraints_num      =  frag_pop.frag_num
        self._lam                 = None

        assert isinstance(frag_pop, FragmentPopulation)
        assert self._nelec_required_list.shape == (self.constraints_num, )
        assert numpy.sum(self._nelec_required_list) <= self._mol.nelectron

    def get_pop_shape(self):
        return self._nelec_required_list.shape

    def make_weight_matrices(self, method="Mulliken"):
        return self._frag_pop.make_weight_matrix_ao(method=method)

    def get_pop_minus_nelec_required(self, weight_matrix, density_matrix=None):
        pop = self._frag_pop.get_pop(weight_matrix, density_matrix=density_matrix, do_spin_pop=False)
        return pop.reshape(self.constraints_num,) - self._nelec_required_list

    def make_cdft_fock_add(self, lam_vals, weight_matrices):
        lam_val_list = numpy.asarray(lam_vals)
        assert lam_val_list.shape == self._nelec_required_list.shape

        fock_add_ao = numpy.einsum("f,fij->ij", lam_vals, weight_matrices)

        if isinstance(self._scf, scf.uhf.UHF):
            tmp_fock_ao = [fock_add_ao, fock_add_ao]
            return numpy.asarray(tmp_fock_ao)
        else:
            tmp_fock_ao = fock_add_ao
            return numpy.asarray(tmp_fock_ao)

    def make_cdft_grad(self, weight_matrix, mo_coeff, mo_occ, mo_energy):
        density_matrix = self._scf.make_rdm1(mo_coeff, mo_occ, mo_energy=mo_energy)
        return self.get_pop_minus_nelec_required(weight_matrix, density_matrix=density_matrix)

    def make_cdft_hess(self, weight_matrix, mo_coeff, mo_occ, mo_energy):
        wao       = weight_matrix
        if mo_coeff.ndim == 3:
            occidxa = numpy.where(mo_occ[0]>0)[0]
            occidxb = numpy.where(mo_occ[1]>0)[0]
            viridxa = numpy.where(mo_occ[0]==0)[0]
            viridxb = numpy.where(mo_occ[1]==0)[0]

            orboa = mo_coeff[0][:,occidxa]
            orbob = mo_coeff[1][:,occidxb]
            orbva = mo_coeff[0][:,viridxa]
            orbvb = mo_coeff[1][:,viridxb]
            e_vo_a = mo_energy[0][occidxa][:,None] - mo_energy[0][viridxa]
            e_vo_b = mo_energy[1][occidxb][:,None] - mo_energy[1][viridxb]
            e_vo   = numpy.asarray([e_vo_a, e_vo_b])

            wvo_a = numpy.einsum("ma,mn,ni->sai", orbva, wao, orboa)
            wvo_b = numpy.einsum("ma,mn,ni->sai", orbvb, wao, orbob)
            wvo   = numpy.asarray([wvo_a, wvo_b])
            hess  = numpy.einsum("smai,snai,sai->mn", wvo, wvo, 1./e_vo)

        elif mo_coeff.ndim == 2:
            occidx = numpy.where(mo_occ==2)[0]
            viridx = numpy.where(mo_occ==0)[0]

            orbv = mo_coeff[:,viridx]
            orbo = mo_coeff[:,occidx]
            e_vo = mo_energy[numpy.where(mo_occ==0)][:,None] - mo_energy[numpy.where(mo_occ==2)]

            wvo  = numpy.einsum("ma,fmn,ni->fai", orbv, wao, orbo)
            hess = 2*numpy.einsum("mai,nai,ai->mn", wvo, wvo, 1./e_vo) 

        return -2*hess

    def make_cdft_hess_exact(self, weight_matrix, mo_coeff, mo_occ, mo_energy):
        wao       = weight_matrix
        mf        = self._scf
        mf.mo_coeff  = mo_coeff
        mf.mo_occ    = mo_occ
        mf.mo_energy = mo_energy
        vresp = mf.gen_response(singlet=None, hermi=1)

        if mo_coeff.ndim == 2:
            occidx = numpy.where(mo_occ==2)[0]
            viridx = numpy.where(mo_occ==0)[0]

            orbv = mo_coeff[:,viridx]
            orbo = mo_coeff[:,occidx]

            nao, nmo = mo_coeff.shape
            nocc = (mo_occ>0).sum()
            nvir = nmo - nocc

            wvo  = numpy.einsum("ma,fmn,ni->fai", orbv, wao, orbo)
            def fvind(x):  # For singlet, closed shell ground state
                dm = reduce(numpy.dot, (orbv, x.reshape(nvir,nocc)*2, orbo.T))
                v1ao = vresp(dm+dm.T)
                return reduce(numpy.dot, (orbv.T, v1ao, orbo)).ravel()

            z = [cphf.solve(fvind, mo_energy, mo_occ, w, max_cycle=10, tol=1e-3)[0] for w in wvo]
            hess = numpy.einsum("fai,gai->fg", z, wvo)
            return 4*hess


def cdft(frag_list, nelec_required_list, init_lam_list=None, pop_scheme="mulliken", 
         alpha=1.0, tol=1e-8, max_iter=200, verbose=4):
    frag_pop = FragmentPopulation(frag_list, verbose=verbose, do_spin_pop=False)
    frag_pop.build()

    cons_pop = Constraints(frag_pop, nelec_required_list)
    wao      = cons_pop.make_weight_matrices(method=pop_scheme)

    mf           = cons_pop._scf
    mf.verbose   = 0
    mf.max_cycle = max_iter

    old_get_fock    = mf.get_fock
    cdft_diis       = scf.diis.SCF_DIIS(mf, mf.diis_file)
    cdft_diis.space = 8

    if init_lam_list is None:
        init_lam_list = numpy.zeros(cons_pop.constraints_num)

    lam_list      = numpy.asarray(init_lam_list)
    
    iter_cdft          = 0
    cdft_maxiter       = 20
    cdft_is_converged  = False
    constraints_tol    = numpy.sqrt(tol)

    e_tot = mf.kernel()
    assert  mf.converged
    dm_last = mf.make_rdm1()

    while not cdft_is_converged and iter_cdft < max_iter:
        def get_fock(h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, mf_diis=None):
            fock_0 = old_get_fock(h1e=h1e, s1e=s1e, vhf=vhf, dm=dm, cycle=-1, diis=None)
            fock_add = cons_pop.make_cdft_fock_add(lam_list, wao)
            fock   = fock_0 + fock_add
            if cycle < 0:
                return fock
            else:
                if s1e is None: s1e = mf.get_ovlp()
                if cycle >= 8:
                    fock = cdft_diis.update(s1e, dm, fock, mf, h1e, vhf)
                return fock
    
        mf.get_fock = get_fock
        e0          = mf.kernel(dm0=dm_last)
        assert  mf.converged
        dm_last = mf.make_rdm1()

        mo_coeff  = mf.mo_coeff
        mo_occ    = mf.mo_occ
        mo_energy = mf.mo_energy

        dm_last = mf.make_rdm1(mo_coeff, mo_occ)
        e_tot   = mf.energy_tot(dm=dm_last)
        grad    = cons_pop.make_cdft_grad(wao, mo_coeff, mo_occ, mo_energy)

        if iter_cdft == 0:
            hess    = cons_pop.make_cdft_hess_exact(wao, mo_coeff, mo_occ, mo_energy)
        else:
            hess    = cons_pop.make_cdft_hess(wao, mo_coeff, mo_occ, mo_energy)

        g_norm     = numpy.linalg.norm(grad)

        lag       = e_tot + numpy.dot(grad, lam_list)
        d_lam     = alpha * get_newton_step_aug_hess(grad, hess)
        dlam_norm = numpy.linalg.norm(d_lam)

        if verbose > 0:
            print()
            print("iter = %d, etot = % 12.8f, lag = % 12.8f, grad = %e, dlam = %e"%(iter_cdft, e_tot, lag, g_norm, dlam_norm))
            print("grad  = ", grad)
            print("lam   = ", lam_list)
            print("dlam  = ", d_lam)

        lam_list   = lam_list + d_lam
        iter_cdft += 1

        cdft_is_converged = g_norm < constraints_tol and dlam_norm < constraints_tol
    return mf

def cdft_inner(frag_list, nelec_required_list, lam_list=None, pop_scheme="mulliken", 
               tol=1e-8, max_iter=200, verbose=0):
    frag_pop = FragmentPopulation(frag_list, verbose=0, do_spin_pop=False)
    frag_pop.build()

    cons_pop = Constraints(frag_pop, nelec_required_list)
    wao      = cons_pop.make_weight_matrices(method=pop_scheme)

    mf           = cons_pop._scf
    mf.verbose   = 0
    mf.max_cycle = max_iter

    old_get_fock    = mf.get_fock
    cdft_diis       = scf.diis.SCF_DIIS(mf, mf.diis_file)
    cdft_diis.space = 8

    lam_list      = numpy.asarray(lam_list)
    
    e_tot = mf.kernel()
    assert  mf.converged
    dm_last = mf.make_rdm1()

    def get_fock(h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, mf_diis=None):
        fock_0 = old_get_fock(h1e=h1e, s1e=s1e, vhf=vhf, dm=dm, cycle=-1, diis=None)
        fock_add = cons_pop.make_cdft_fock_add(lam_list, wao)
        fock   = fock_0 + fock_add
        if cycle < 0:
            return fock
        else:
            if s1e is None: s1e = mf.get_ovlp()
            if cycle >= 8:
                fock = cdft_diis.update(s1e, dm, fock, mf, h1e, vhf)
            return fock

    mf.get_fock = get_fock
    e0          = mf.kernel(dm0=dm_last)
    assert  mf.converged

    mo_coeff  = mf.mo_coeff
    mo_occ    = mf.mo_occ
    mo_energy = mf.mo_energy

    dm      = mf.make_rdm1(mo_coeff, mo_occ)
    e_tot   = mf.energy_tot(dm=dm)

    grad     = cons_pop.make_cdft_grad(wao, mo_coeff, mo_occ, mo_energy)
    lag      = e_tot + numpy.dot(grad, lam_list)
    hess       = cons_pop.make_cdft_hess(wao, mo_coeff, mo_occ, mo_energy)
    exact_hess = cons_pop.make_cdft_hess_exact(wao, mo_coeff, mo_occ, mo_energy)
    return e_tot, lag, grad, hess, exact_hess

if __name__ == '__main__':
    mol1 = gto.Mole()
    mol1.atom = '''
    O        -1.5191160500    0.1203799285    0.0000000000
    H        -1.9144370615   -0.7531546521    0.0000000000
    H        -0.5641810316   -0.0319363903    0.0000000000'''
    mol1.basis = 'cc-pvtz'
    mol1.build()
    mol2 = gto.Mole()
    mol2.atom = '''
    O         1.3908332592   -0.1088624008    0.0000000000
    H         1.7527099026    0.3464799298   -0.7644430086
    H         1.7527099026    0.3464799298    0.7644430086'''
    mol2.basis = 'cc-pvtz'
    mol2.build()

    frag1 = scf.RHF(mol1)
    frag1.kernel()

    frag2 = scf.RHF(mol2)
    frag2.kernel()

    frag_list = [frag1, frag2]
    fp = FragmentPopulation(frag_list, verbose=0)
    fp.build()
    e0 = fp._scf.energy_tot()
    
    dm_last = cdft(frag_list, [9.7, 10.3], init_lam_list=None, pop_scheme="mul")
    wao = fp.make_weight_matrix_ao(method="mul")
    pop = fp.get_pop(wao, density_matrix=dm_last)
    print("pop = ", pop)
    etot = fp._scf.energy_tot(dm=dm_last)
    print("etot = ", etot)
    print("dene = ", etot-e0)