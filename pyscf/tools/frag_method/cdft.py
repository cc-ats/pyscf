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
from pyscf.dft import numint
from frag_pop import FragmentPopulation

from copy import deepcopy
from functools import reduce

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
        assert numpy.sum(self._nelec_required_list) == self._mol.nelectron

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
            hess = numpy.einsum("mai,nai,ai->mn", wvo, wvo, 1./e_vo) 

        return hess

def cdft(frag_list, nelec_required_list, init_lam_list=None, pop_scheme="mulliken", 
                                         alpha=0.2,          tol=1e-5,        constraints_tol=1e-3, 
                                         maxiter=200,        diis_pos='post', diis_type=1, verbose=4):
    frag_pop = FragmentPopulation(frag_list, verbose=verbose, do_spin_pop=False)
    frag_pop.build()
    cons_pop = Constraints(frag_pop, nelec_required_list)
    cons_pop._lam = numpy.array(init_lam_list)

    mf           = cons_pop._scf
    mf.verbose   = verbose
    mf.max_cycle = maxiter

    old_get_fock = mf.get_fock

    if init_lam_list is None:
        init_lam_list = numpy.zeros(cons_pop.constraints_num)
    
    cdft_diis = lib.diis.DIIS()
    cdft_diis.space = 8

    lam_list = numpy.asarray(init_lam_list)

    wao = cons_pop.make_weight_matrices(method=pop_scheme)

    def get_fock(h1e, s1e, vhf, dm, cycle=0, mf_diis=None):
        fock_0   = old_get_fock(h1e, s1e, vhf, dm, cycle, None)
        lam_list = cons_pop._lam
        if mf_diis is None:
            fock_add = cons_pop.make_cdft_fock_add(lam_list, wao)
            return fock_0 + fock_add

        cdft_conv_flag = False
        if cycle < 10:
            inner_max_cycle = 20
        else:
            inner_max_cycle = 50

        if verbose > 3:
            print("\nCDFT INNER LOOP:")

        fock_0   = old_get_fock(h1e, s1e, vhf, dm, cycle, None)
        fock_add = cons_pop.make_cdft_fock_add(lam_list, wao)
        fock = fock_0 + fock_add #ZHC

        if diis_pos == 'pre' or diis_pos == 'both':
            for it in range(inner_max_cycle): # TO BE MODIFIED
                fock_add = get_fock_add_cdft(constraints, V_0, C_inv)
                fock = fock_0 + fock_add #ZHC

                mo_energy, mo_coeff = mf.eig(fock, s1e)
                mo_occ = mf.get_occ(mo_energy, mo_coeff)

                # Required by hess_cdft function
                mf.mo_energy = mo_energy
                mf.mo_coeff = mo_coeff
                mf.mo_occ = mo_occ

                if lo_method.lower() == 'iao':
                    mo_on_loc_ao = get_localized_orbitals(mf, lo_method, mo_coeff)[1]
                else:
                    mo_on_loc_ao = np.einsum('...jk,...kl->...jl', C_inv, mo_coeff)

                orb_pop = pop_analysis(mf, mo_on_loc_ao, disp=False)
                W_new = W_cdft(mf, constraints, V_0, orb_pop)
                jacob, N_cur = jac_cdft(mf, constraints, V_0, orb_pop)
                hess = hess_cdft(mf, constraints, V_0, mo_on_loc_ao)

                deltaV = get_newton_step_aug_hess(jacob,hess)
                #deltaV = np.linalg.solve (hess, -jacob)

                if it < 5 :
                    stp = min(0.05, alpha*0.1)
                else:
                    stp = alpha

                V = V_0 + deltaV * stp
                g_norm = np.linalg.norm(jacob)
                if verbose > 3:
                    print("  loop %4s : W: %.5e    V_c: %s     Nele: %s      g_norm: %.3e    "
                          % (it,W_new, V_0, N_cur, g_norm))
                if g_norm < tol and np.linalg.norm(V-V_0) < constraints_tol:
                    cdft_conv_flag = True
                    break
                V_0 = V

        if cycle > 1:
            if diis_type == 1:
                fock = cdft_diis.update(fock_0, scf.diis.get_err_vec(s1e, dm, fock)) + fock_add
            elif diis_type == 2:
                # TO DO difference < threshold...
                fock = cdft_diis.update(fock)
            elif diis_type == 3:
                fock = cdft_diis.update(fock, scf.diis.get_err_vec(s1e, dm, fock))
            else:
                print("\nWARN: Unknow CDFT DIIS type, NO DIIS IS USED!!!\n")

        if diis_pos == 'post' or diis_pos == 'both':
            cdft_conv_flag = False
            fock_0 = fock - fock_add
            for it in range(inner_max_cycle): # TO BE MODIFIED
                fock_add = get_fock_add_cdft(constraints, V_0, C_inv)
                fock = fock_0 + fock_add #ZHC

                mo_energy, mo_coeff = mf.eig(fock, s1e)
                mo_occ = mf.get_occ(mo_energy, mo_coeff)

                # Required by hess_cdft function
                mf.mo_energy = mo_energy
                mf.mo_coeff = mo_coeff
                mf.mo_occ = mo_occ

                if lo_method.lower() == 'iao':
                    mo_on_loc_ao = get_localized_orbitals(mf, lo_method, mo_coeff)[1]
                else:
                    mo_on_loc_ao = np.einsum('...jk,...kl->...jl', C_inv, mo_coeff)

                orb_pop = pop_analysis(mf, mo_on_loc_ao, disp=False)
                W_new = W_cdft(mf, constraints, V_0, orb_pop)
                jacob, N_cur = jac_cdft(mf, constraints, V_0, orb_pop)
                hess = hess_cdft(mf, constraints, V_0, mo_on_loc_ao)
                deltaV = np.linalg.solve (hess, -jacob)

                if it < 5 :
                    stp = min(0.05, alpha*0.1)
                else:
                    stp = alpha

                V = V_0 + deltaV * stp
                g_norm = np.linalg.norm(jacob)
                if verbose > 3:
                    print("  loop %4s : W: %.5e    V_c: %s     Nele: %s      g_norm: %.3e    "
                          % (it,W_new, V_0, N_cur, g_norm))
                if g_norm < tol and np.linalg.norm(V-V_0) < constraints_tol:
                    cdft_conv_flag = True
                    break
                V_0 = V

        if verbose > 0:
            print("CDFT W: %.5e   g_norm: %.3e    "%(W_new, g_norm))

        constraints._converged = cdft_conv_flag
        constraints._final_V = V_0
        return fock

if __name__ == '__main__':
    mol1 = gto.Mole()
    mol1.atom = '''
    O        -1.5191160500    0.1203799285    0.0000000000
    H        -1.9144370615   -0.7531546521    0.0000000000
    H        -0.5641810316   -0.0319363903    0.0000000000'''
    mol1.basis = '6-31g(d)'
    mol1.build()
    mol2 = gto.Mole()
    mol2.atom = '''
    O         1.3908332592   -0.1088624008    0.0000000000
    H         1.7527099026    0.3464799298   -0.7644430086
    H         1.7527099026    0.3464799298    0.7644430086'''
    mol2.basis = '6-31g(d)'
    mol2.build()

    frag1 = scf.RHF(mol1)
    frag1.kernel()

    frag2 = scf.RHF(mol2)
    frag2.kernel()

    frag_list = [frag1, frag2]
    fp = FragmentPopulation(frag_list, verbose=0)
    fp.build()
    
    pop_cons = Constraints(fp, [10, 10])
    wao      = pop_cons.make_weight_matrices(method="mul")
    mo_coeff  = pop_cons._scf.mo_coeff
    mo_occ    = pop_cons._scf.mo_occ
    mo_energy = pop_cons._scf.mo_energy

    grad = pop_cons.make_cdft_grad(wao, mo_coeff, mo_occ, mo_energy)
    hess = pop_cons.make_cdft_hess(wao, mo_coeff, mo_occ, mo_energy)
    print("delta V = \n", get_newton_step_aug_hess(grad, hess))