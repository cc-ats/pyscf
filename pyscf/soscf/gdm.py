import sys
from sys import stdout
import time
from functools import reduce

import numpy
import scipy
from numpy        import dot
from scipy.linalg import expm, orth
from numpy.linalg import svd

from pyscf     import gto
from pyscf     import lib
from pyscf.lib import logger
from pyscf.tools.dump_mat import dump_rec

from pyscf.scf import chkfile
from pyscf.scf import addons
from pyscf.scf import hf, uhf

from pyscf import __config__

from .optimizer import LineSearch, DogLegSearch

SVD_TOL               = getattr(__config__, 'soscf_gdm_effective_svd_tol',   1e-5)
DEF_EFF_THRSH         = getattr(__config__, 'soscf_gdm_def_eff_thrsh',    -1000.0)
CURV_CONDITION_SCALE  = getattr(__config__, 'soscf_gdm_curv_condition_scale', 0.9)
PREC_MIN              = getattr(__config__, 'soscf_gdm_prec_min',            0.05)

def expmat(a):
    return expm(a)

def kernel(mf, mo_coeff=None, mo_occ=None, dm=None,
           conv_tol=1e-10, conv_tol_grad=None, max_cycle=50, dump_chk=True,
           callback=None, verbose=logger.NOTE):
    cput0 = (time.clock(), time.time())
    log   = logger.new_logger(mf, verbose)
    mol   = mf._scf.mol
    if mol != mf.mol:
        logger.warn(mf, 'dual-basis SOSCF is an experimental feature. It is '
                    'still in testing.')

    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(conv_tol)
        log.info('Set conv_tol_grad to %g', conv_tol_grad)

    h1e = mf._scf.get_hcore(mol)
    s1e = mf._scf.get_ovlp(mol)

    if mo_coeff is not None and mo_occ is not None:
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        vhf = mf._scf.get_veff(mol, dm)
        fock = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
    else:
        if dm is None:
            logger.debug(mf, 'Initial guess density matrix is not given. '
                         'Generating initial guess from %s', mf.init_guess)
            dm = mf.get_init_guess(mf._scf.mol, mf.init_guess)
        vhf = mf._scf.get_veff(mol, dm)
        fock = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm, dm_last = mf.make_rdm1(mo_coeff, mo_occ), dm
        vhf = mf._scf.get_veff(mol, dm, dm_last=dm_last, vhf_last=vhf)

    # Save mo_coeff and mo_occ because they are needed by function rotate_mo
    mf.mo_coeff, mf.mo_occ = mo_coeff, mo_occ

    e_tot = mf._scf.energy_tot(dm, h1e, vhf)
    fock = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
    log.info('Initial guess E= %.15g  |g|= %g', e_tot,
             numpy.linalg.norm(mf._scf.get_grad(mo_coeff, mo_occ, fock)))

    if dump_chk and mf.chkfile:
        chkfile.save_mol(mol, mf.chkfile)

    # Copy the integral file to soscf object to avoid the integrals being cached
    # twice.
    if mol is mf.mol and not getattr(mf, 'with_df', None):
        mf._eri = mf._scf._eri

    scf_conv         = False
    is_first_step    = True
    iter_gdm         = 0
    iter_line_search = 0

    eff_thresh       = -100.00

    line_search_obj    = LineSearch()
    dog_leg_search_obj = DogLegSearch()

    grad_list    = None
    step_list    = None

    num_subspace_vec = None

    gdm_orb_step     = None

    ene_cur   = e_tot
    ene_pre   = None
    g_pre     = None
    g_cur     = None
    alpha_pre = None
    alpha_cur = None

    cur_mo_coeff = mf.mo_coeff
    pre_mo_coeff = None

    cput1 = log.timer('initializing second order scf', *cput0)

    while not scf_conv and iter_gdm < mf.gdm_max_cycle:
        if verbose is None:
            verbose = mf.verbose

        log = logger.new_logger(mf, verbose)
        log.note('\n')
        log.info('Geometric Direct Minimization')

        dm   = mf.make_rdm1(cur_mo_coeff, mo_occ)
        dm   = lib.tag_array(dm, mo_coeff=cur_mo_coeff, mo_occ=mo_occ)
        vhf  = mf.get_veff(mol, dm, dm_last, vhf)
        fock = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
        coo, cvv, cur_mo_coeff, mo_energy = mf.get_canonicalize(cur_mo_coeff, mo_occ, fock_ao=fock)
        mo_occ = mo_occ = mf.get_occ(mo_energy=mo_energy, mo_coeff=cur_mo_coeff)

        grad, diag_hess = mf.get_orb_grad_hess(cur_mo_coeff, mo_occ, fock_ao=fock)
        prec      = 1.0/numpy.sqrt(diag_hess)

        norm_gorb = numpy.linalg.norm(grad)
        norm_gorb = norm_gorb / numpy.sqrt(norm_gorb.size)

        if (norm_gorb < mf.gdm_conv_tol * mf.gdm_conv_tol):
            log.debug("---------- Successful GDM Step ----------")
            log.debug("Hopefully we are in the right direction.")
            scf_conv = True

        if gdm_orb_step is None:
            gdm_orb_step = numpy.empty_like(grad)

        if grad_list is None:
            grad_list = []
            grad_list.append(grad)
        else:
            num_grad = len(grad_list)
            if (num_subspace_vec == num_grad):
                grad_list.append(grad)
            elif num_subspace_vec < num_grad:
                grad_list[num_subspace_vec] = grad
            else:
                RuntimeError("GDM dimensioning error!")

        if not is_first_step:
            diff_ene = ene_cur - ene_pre
            trust_radius = dog_leg_search_obj.get_trust_radius(diff_ene)
            if - diff_ene >= 1e-4 * g_pre:
                is_wolfe1 = True
            elif numpy.abs(diff_ene) <= eff_thresh:
                is_wolfe1 = True
            else:
                is_wolfe1 = False
            
            g_cur      = numpy.dot(gdm_orb_step, grad)
            xx         = numpy.dot(gdm_orb_step, gdm_orb_step)
            x_cur      = numpy.sqrt(xx)
            g_cur      = g_cur/x_cur

            if g_cur >= CURV_CONDITION_SCALE * g_pre:
                is_wolfe2 = True
                if iter_line_search > mf.max_line_search:
                    is_wolfe2     = True
                    is_first_step = True

        if is_wolfe1 and is_wolfe2:
            if grad_list is not None:
                grad_list = mf.update_grad_list(coo, cvv, grad_list)
            if step_list is not None:
                step_list = mf.update_grad_list(coo, cvv, step_list)

            iter_line_search = 0
            mf.mo_coeff = mo_coeff

            num_subspace_vec += 1
            dim_subspace  = -1

            subspace_mat, ene_weight_subspace = mf.proj_ene_weight_subspace(grad_list,step_list,prec)
            hess_inv = mf.get_hess_inv(subspace_mat)
            tmp_step_vec = dog_leg_search_obj.next_step(subspace_mat, hess_inv)
            gdm_orb_step = numpy.dot(ene_weight_subspace, tmp_step_vec)
        else:
            mo_coeff = mf.mo_coeff
            iter_line_search += 1
            alpha_cur = line_search_obj.next_step(ene_pre, alpha_pre, g_pre,
                                                  ene_cur, alpha_cur, g_cur,
                                                  is_wolfe1, is_wolfe2)
            gdm_orb_step *= alpha_cur/numpy.sqrt(xx)
            step_max = numpy.max(gdm_orb_step)

        grad_prec  = grad * prec
        err        = numpy.max(numpy.abs(grad_prec))

        print("num_subspace_vec = %d"%num_subspace_vec)
        if step_list is None:
            step_list = []
            step_list.append(gdm_orb_step)
        else:
            step_list.append(gdm_orb_step)

        is_first_step = False

        pre_mo_coeff = cur_mo_coeff
        cur_mo_coeff = mf.update_mo_coeff(gdm_orb_step, pre_mo_coeff, mo_occ)

        ene_pre       = ene_cur
        dm_last       = dm


# returns an exact gradient and approximate hessian
def get_orb_grad_hess_rhf(mf, mo_coeff, mo_occ, fock_ao=None):
    if fock_ao is None:
        dm      = mf.make_rdm1(mo_coeff, mo_occ)
        fock_ao = mf.get_fock(dm=dm)
    mol = mf.mol

    occidx = numpy.where(mo_occ==2)[0]
    viridx = numpy.where(mo_occ==0)[0]
    nocc = len(occidx)
    nvir = len(viridx)
    orbo = mo_coeff[:,occidx]
    orbv = mo_coeff[:,viridx]

    f1   = reduce(numpy.dot, (mo_coeff.conj().T, fock_ao, mo_coeff))
    foo  = f1[occidx[:,None],occidx]
    fvv  = f1[viridx[:,None],viridx]

    grad   = f1[viridx[:,None],occidx] * 2
    h_diag = (fvv.diagonal().real[:,None] - foo.diagonal().real) * 2
    
    diff_min = 1.0
    err = numpy.linalg.norm(grad)/grad.size
    if err < 1.0/8.0:
        diff_min = 1.0/16.0
    elif err < 8.0:
        diff_min = 1.0/4.0

    abs_h_diag = numpy.abs(h_diag.reshape(-1))
    abs_h_diag[abs_h_diag<diff_min] = diff_min

    return grad.reshape(-1), abs_h_diag

def get_canonicalize_rhf(mf, mo_coeff, mo_occ, fock_ao=None):
    if fock_ao is None:
        dm      = mf.make_rdm1(mo_coeff, mo_occ)
        fock_ao = mf.get_fock(dm=dm)

    occidx  = mo_occ == 2
    viridx  = mo_occ == 0

    nao, nmo      = mo_coeff.shape
    mo_coeff_cano = numpy.empty_like(mo_coeff)
    mo_energy     = numpy.empty(nmo)

    orbo    = mo_coeff[:,occidx]
    foo     = reduce(numpy.dot, (orbo.conj().T, fock_ao, orbo))
    e, cano_trans_orb_oo    = scipy.linalg.eig(foo)
    mo_coeff_cano[:,occidx] = numpy.dot(orbo, cano_trans_orb_oo)
    mo_energy[occidx]       = e

    orbv   = mo_coeff[:,viridx]
    fvv    = reduce(numpy.dot, (orbv.conj().T, fock_ao, orbv))
    e, cano_trans_orb_vv    = scipy.linalg.eig(fvv)
    mo_coeff_cano[:,viridx] = numpy.dot(orbv, cano_trans_orb_vv)
    mo_energy[viridx]       = e

    return cano_trans_orb_oo, cano_trans_orb_vv, mo_coeff_cano, mo_energy

def update_grad_list_rhf(mf, coo, cvv, grad_list):
    tmp_grad_list = numpy.einsum("ab,tai,ij->tbj", cvv.T, grad_list, coo)
    return list(tmp_grad_list)

def update_mo_coeff_rhf(mf, gdm_orb_step, mo_coeff, mo_occ):
    nmo     = len(mo_occ)
    occidxa = mo_occ>0
    occidxb = mo_occ==2
    viridxa = ~occidxa
    viridxb = ~occidxb
    idx     = (viridxa[:,None] & occidxa) | (viridxb[:,None] & occidxb)

    dtheta      = numpy.zeros((nmo,nmo), dtype=gdm_orb_step.dtype)
    dtheta[idx] = gdm_orb_step
    u           = expm(dtheta)

    if mo_coeff is not None:
        u = numpy.dot(mo_coeff, u)
    tmp_mo_coeff = u
    return tmp_mo_coeff

class GDMOptimizer(object):
    '''
    Attributes for GDM solver:
        max_cycle_inner : int
            gdm iterations within eacy macro iterations.
            Default is 10
        max_stepsize : int
            The step size for orbital rotation.  Small step is prefered.
            Default is 0.05.
        canonicalization : bool
            To control whether to canonicalize the orbitals optimized by
            GDM solver.
            Default is True.
    '''

    gdm_conv_tol      = getattr(__config__, 'soscf_gdm_conv_tol', 1e-6)
    max_cycle_gdm     = getattr(__config__, 'soscf_gdm_max_cycle',   40)
    max_cycle_ls      = getattr(__config__, 'soscf_gdm_max_cycle_inner'   , 12)
    max_stepsize_ls   = getattr(__config__, 'soscf_gdm_max_stepsize',      .05)
    max_subspace_size = getattr(__config__, 'soscf_gdm_max_subspace_size',  20)
    

    def __init__(self, mf):
        self.__dict__.update(mf.__dict__)
        self._scf = mf

        self._keys.update((   'max_cycle_ls',
                           'max_stepsize_ls',
                         'max_subspace_size',
                              'gdm_conv_tol',
                             'max_cycle_gdm'))

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        self._scf.dump_flags(verbose)
        log.info('******** %s GDM solver flags ********', self._scf.__class__)
        return self

    def proj_ene_weight_subspace(self, grad_list, step_list, prec):
        num_scale_prec   = 0
        num_subspace_vec = len(grad_list)
        
        prec_inv         = numpy.empty_like(prec)
        for i, p in enumerate(prec):
            prec_inv[i] = 1.0/numpy.abs(p)
            if prec_inv[i] < PREC_MIN:
                num_scale_prec += 1
                prec_inv[i]     = PREC_MIN
        prec     = 1.0/prec_inv

        cur_norm = numpy.linalg.norm(grad_list[-1])
        for i in range(num_subspace_vec):
            igrad    = num_subspace_vec - i - 1
            grad     = grad_list[igrad]
            tmp_vec  = grad * prec
            tmp_norm = numpy.linalg.norm(tmp_vec)
            if tmp_norm > cur_norm*1e10 or (igrad==1 and num_subspace_vec==self.max_subspace_size):
                del grad_list[0:igrad]
                del step_list[0:igrad]
                num_subspace_vec = num_subspace_vec - i
                break
        
        ene_weight_subspace_1    = [grad*prec     for grad in grad_list]
        ene_weight_subspace_2    = [step*prec_inv for step in step_list]
        ene_weight_subspace      = numpy.asarray(ene_weight_subspace_1 + ene_weight_subspace_2)

        ew_ss_t_dot_ew_ss = dot(ene_weight_subspace.T, ene_weight_subspace)
        tmp_ew_ss = numpy.copy(ene_weight_subspace)
        
        nrow, ncol = ew_ss_t_dot_ew_ss.shape
        tmp_diag = numpy.diag(ew_ss_t_dot_ew_ss)
        for i in range(nrow):
            for j in range(i+1,ncol):
                tmp = ew_ss_t_dot_ew_ss[i,j]/numpy.sqrt(tmp_diag[i]*tmp_diag[j])
                ew_ss_t_dot_ew_ss[i,j] = tmp
                ew_ss_t_dot_ew_ss[j,i] = tmp
            ew_ss_t_dot_ew_ss[i,i] = 1.0
        
        eig_vals, eig_vecs = numpy.linalg.eigh(ew_ss_t_dot_ew_ss)
        sort_idx           = eig_vals.argsort()
        sort_eig_vals      = numpy.abs(eig_vals[sort_idx])
        tmp_eig_vals       = sort_eig_vals/sort_eig_vals[-1]
        sort_eig_vecs      = eig_vecs[:, sort_idx]
        
        tmp_idx            = tmp_eig_vals>1e-8
        tmp_vals           = sort_eig_vals[tmp_idx]
        inv_sqrt_vals      = 1/numpy.sqrt(tmp_vals)
        tmp_vecs           = sort_eig_vecs[:,tmp_idx]
        
        vecs_dot_inv_sqrt_vals = dot(tmp_vecs, numpy.diag(inv_sqrt_vals))
        
        for i in range(ncol):
            tmp_norm = numpy.linalg.norm(tmp_ew_ss[:,i])
            tmp_ew_ss[:,i] = tmp_ew_ss[:,i]/tmp_norm
        
        tmp_mat = dot(tmp_ew_ss, vecs_dot_inv_sqrt_vals)
        for i in range(tmp_vals.size):
            tmp_norm = numpy.linalg.norm(tmp_mat[:,i])
            tmp_mat[:,i] = tmp_mat[:,i]/tmp_norm
        
        subspace_mat = dot(ene_weight_subspace.T, tmp_mat)
        return subspace_mat, tmp_mat

    def get_hess_inv(self, subspace_mat):
        dim_subspace, tmp_num = subspace_mat.shape
        num_subspace_vec = (tmp_num+1)//2
        gmat = subspace_mat[:,:num_subspace_vec]
        smat = subspace_mat[:,num_subspace_vec:(2*num_subspace_vec)]
        
        hess_inv = numpy.eye(dim_subspace)
        for i_subspace_vec in range(1,num_subspace_vec):
            g_new = gmat[:,i_subspace_vec]
            g_old = gmat[:,i_subspace_vec-1]
            
            yk    = (g_new - g_old).reshape(4,1)
            sk    = (smat[:,i_subspace_vec-1]).reshape(4,1)
            
            yk_dot_sk = dot(yk.T,sk)
            sk_dot_sk = dot(sk.T,sk)
            yk_dot_yk = dot(yk.T,yk)
            inv_hess_thresh = numpy.sqrt(sk_dot_sk) * numpy.sqrt(yk_dot_yk)
            
            if numpy.abs(yk_dot_sk) >= 1e-8 * inv_hess_thresh:
                u = numpy.eye(dim_subspace) - dot(sk, yk.T)/yk_dot_sk
                hess_inv = reduce(dot, [u, hess_inv, u.T]) + dot(sk,sk.T)/yk_dot_sk
        
        eig_vals, eig_vecs = numpy.linalg.eigh(hess_inv)
        abs_eig_vals       = numpy.abs(eig_vals)
        hess_inv           = reduce(dot, [eig_vecs, abs_eig_vals, eig_vecs.T])
        return hess_inv
    
    def kernel(self):
        pass

    get_orb_grad_hess = get_orb_grad_hess_rhf
    get_canonicalize  = get_canonicalize_rhf
    update_grad_list  = update_grad_list_rhf
    update_mo_coeff   = update_mo_coeff_rhf

def gdm(mf):
    from pyscf import scf

    if isinstance(mf, GDMOptimizer):
        return mf

    assert(isinstance(mf, hf.SCF))
    if mf.__doc__ is None:
        mf_doc = ''
    else:
        mf_doc = mf.__doc__

    if isinstance(mf, uhf.UHF):
        class SecondOrderUHF(GDMOptimizer, mf.__class__):
            __doc__ = mf_doc + GDMOptimizer.__doc__

            gen_grad_and_hess = gen_grad_and_hess_uhf

            def update_rotate_matrix(self, dx, mo_occ, u0=1, mo_coeff=None):
                occidxa = mo_occ[0] > 0
                occidxb = mo_occ[1] > 0
                viridxa = ~occidxa
                viridxb = ~occidxb

                nmo = len(occidxa)
                dr = numpy.zeros((2,nmo,nmo), dtype=dx.dtype)
                uniq = numpy.array((viridxa[:,None] & occidxa,
                                    viridxb[:,None] & occidxb))
                dr[uniq] = dx
                dr = dr - dr.conj().transpose(0,2,1)

                if isinstance(u0, int) and u0 == 1:
                    return numpy.asarray((expmat(dr[0]), expmat(dr[1])))
                else:
                    return numpy.asarray((numpy.dot(u0[0], expmat(dr[0])),
                                          numpy.dot(u0[1], expmat(dr[1]))))

            def rotate_mo(self, mo_coeff, u, log=None):
                mo = numpy.asarray((numpy.dot(mo_coeff[0], u[0]),
                                    numpy.dot(mo_coeff[1], u[1])))
                if self._scf.mol.symmetry:
                    orbsym = uhf_symm.get_orbsym(self._scf.mol, mo_coeff)
                    mo = lib.tag_array(mo, orbsym=orbsym)
                return mo

            def spin_square(self, mo_coeff=None, s=None):
                if mo_coeff is None:
                    mo_coeff = (self.mo_coeff[0][:,self.mo_occ[0]>0],
                                self.mo_coeff[1][:,self.mo_occ[1]>0])
                return self._scf.spin_square(mo_coeff, s)

            def kernel(self, mo_coeff=None, mo_occ=None, dm0=None):
                if isinstance(mo_coeff, numpy.ndarray) and mo_coeff.ndim == 2:
                    mo_coeff = (mo_coeff, mo_coeff)
                if isinstance(mo_occ, numpy.ndarray) and mo_occ.ndim == 1:
                    mo_occ = (numpy.asarray(mo_occ >0, dtype=numpy.double),
                              numpy.asarray(mo_occ==2, dtype=numpy.double))
                return GDMOptimizer.kernel(self, mo_coeff, mo_occ, dm0)

        return SecondOrderUHF(mf)

    else:
        class SecondOrderRHF(GDMOptimizer, mf.__class__):
            __doc__ = mf_doc + GDMOptimizer.__doc__
            gen_grad_and_hess = get_orb_grad_hess_rhf
        return SecondOrderRHF(mf)
