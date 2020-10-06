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
DIAG_SQRT_H_MIN       = getattr(__config__, 'soscf_gdm_diag_sqrt_h_min',     0.05)

MAX_CYCLE_GDM         = getattr(__config__, 'soscf_gdm_max_cycle',          40)
MAX_CYCLE_LS          = getattr(__config__, 'soscf_gdm_max_cycle_inner',    12)
MAX_STEPSIZE_LS       = getattr(__config__, 'soscf_gdm_max_stepsize',      .05)
MAX_SUBSPACE_SIZE     = getattr(__config__, 'soscf_gdm_max_subspace_size',  20)

TIGHT_GRAD_CONV_TOL   = getattr(__config__, 'scf_hf_kernel_tight_grad_conv_tol', True)

def expmat(a):
    return expm(a)

def kernel(mf, mo_coeff=None, mo_occ=None, dm=None,
           conv_tol=1e-10, conv_tol_grad=1e-6,
           conv_check=True,
           max_cycle=50, dump_chk=True,
           callback=None, verbose=logger.NOTE):
    cput0 = (time.clock(), time.time())
    log   = logger.new_logger(mf, verbose)
    mol   = mf._scf.mol
    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(conv_tol)
        logger.info(mf, 'Set conv_tol_grad to %g', conv_tol_grad)

    h1e = mf._scf.get_hcore(mol)
    s1e = mf._scf.get_ovlp(mol)
    cond = lib.cond(s1e)
    logger.debug(mf, 'cond(S) = %s', cond)
    if numpy.max(cond)*1e-17 > conv_tol:
        logger.warn(mf, 'Singularity detected in overlap matrix (condition number = %4.3g). '
                    'SCF may be inaccurate and hard to converge.', numpy.max(cond))

    if mo_coeff is not None and mo_occ is not None:
        dm_last = None
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        vhf = mf._scf.get_veff(mol, dm)
        fock = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
    else:
        if dm is None:
            logger.debug(mf, 'Initial guess density matrix is not given. '
                         'Generating initial guess from %s', mf.init_guess)
            dm = mf.get_init_guess(mf._scf.mol, mf.init_guess)
        dm_last = None
        vhf = mf._scf.get_veff(mol, dm)
        fock = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm, dm_last = mf.make_rdm1(mo_coeff, mo_occ), dm
        vhf = mf._scf.get_veff(mol, dm, dm_last=dm_last, vhf_last=vhf)

    mf.mo_coeff, mf.mo_occ = mo_coeff, mo_occ
    e_tot = mf._scf.energy_tot(dm, h1e, vhf)
    fock  = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
    log.info('Initial guess E= %.15g  |g|= %g', e_tot,
             numpy.linalg.norm(mf._scf.get_grad(mo_coeff, mo_occ, fock)))

    if dump_chk and mf.chkfile:
        chkfile.save_mol(mol, mf.chkfile)

    if mol is mf.mol and not getattr(mf, 'with_df', None):
        mf._eri = mf._scf._eri

    scf_conv         = False
    is_first_step    = True
    is_wolfe1        = True
    is_wolfe2        = True

    iter_gdm         = 0
    iter_line_search = 0

    line_search_obj    = LineSearch()
    dog_leg_search_obj = DogLegSearch()

    mf._grad_list    = None
    mf._step_list    = None

    num_subspace_vec = 0
    dim_subspace     = None
    gdm_orb_step     = None

    ene_cur   = e_tot
    ene_pre   = None

    g_pre     = None
    g_cur     = None
    
    alpha_pre = None
    alpha_cur = None

    cur_mo_coeff = mf.mo_coeff 
    pre_mo_coeff = None
    step_max     = None

    cput1 = log.timer('Initializing GDM', *cput0)

    while not scf_conv and iter_gdm < max_cycle:
        vhf                 = mf.get_veff(mol, dm, dm_last, vhf)
        fock                = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
        ene_cur             = mf.energy_tot(dm, h1e, vhf)
        coo, cvv, mo_coeff_cano, mo_energy = mf.get_canonicalize(cur_mo_coeff, mo_occ, fock_ao=fock)
        print("cur_mo_coeff = \n")
        dump_rec(stdout, cur_mo_coeff)
        print("mo_coeff_cano = \n")
        dump_rec(stdout, mo_coeff_cano)

        grad, diag_hess = mf.get_orb_grad_hess(cur_mo_coeff, coo, cvv, mo_occ, fock_ao=fock)
        cur_mo_coeff    = mo_coeff_cano
        diag_sqrt_h_inv = 1.0/numpy.sqrt(diag_hess)
        norm_gorb       = numpy.linalg.norm(grad)/numpy.sqrt(grad.size)

        if (norm_gorb < conv_tol_grad):
            scf_conv = True
            break

        if gdm_orb_step is None:
            gdm_orb_step = numpy.empty_like(grad)

        if mf._grad_list is None:
            mf._grad_list = []
            mf._grad_list.append(grad)
        else:
            num_grad = len(mf._grad_list)
            if (num_subspace_vec == num_grad):
                mf._grad_list.append(grad)
            elif num_subspace_vec < num_grad:
                mf._grad_list[num_subspace_vec] = grad
            else:
                RuntimeError("GDM dimensioning error!")

        if not is_first_step:
            diff_ene     = ene_cur - ene_pre
            print("diff_ene = %f"%diff_ene)
            log.info('cycle= %d E= %.15g  delta_E= %4.3g  |g|= %4.3g  step_max= %4.3g',iter_gdm, ene_cur, diff_ene, norm_gorb, step_max)
            trust_radius = dog_leg_search_obj.get_trust_radius(diff_ene)
            if - diff_ene >= 1e-4 * g_pre:
                is_wolfe1 = True
            else:
                is_wolfe1 = False
            
            print("gdm_orb_step = \n", gdm_orb_step)
            print("grad = \n", grad)
            g_cur      = numpy.dot(gdm_orb_step, grad)
            alpha2     = numpy.dot(gdm_orb_step, gdm_orb_step)
            alpha      = numpy.sqrt(alpha2)
            alpha_cur  = alpha
            print("g_cur = %f"%g_cur)
            g_cur      = g_cur/alpha_cur
            print("g_cur = %f"%g_cur)
            if g_cur >= CURV_CONDITION_SCALE * g_pre:
                is_wolfe2 = True
                if iter_line_search > mf.max_cycle_ls:
                    is_first_step = True
            else:
                is_wolfe2 = False
                if numpy.abs(step_max) > 0.1:
                    is_wolfe2 = True
                    if iter_line_search > mf.max_cycle_ls:
                        is_first_step = True

        if is_wolfe1 and is_wolfe2:
            log.debug("Dogleg search!")
            if mf._grad_list is not None:
                mf._grad_list = mf.update_grad_list(coo, cvv, mf._grad_list)
            if mf._step_list is not None:
                mf._step_list = mf.update_grad_list(coo, cvv, mf._step_list)

            mf.mo_coeff  = cur_mo_coeff
            subspace, ew_subspace = mf.proj_ew_subspace(diag_sqrt_h_inv)

            tmp_step     = dog_leg_search_obj.next_step(subspace)
            tmp_step     = numpy.dot(ew_subspace, tmp_step)
            gdm_orb_step = -tmp_step * diag_sqrt_h_inv
            step_max     = numpy.max(numpy.abs(gdm_orb_step))


            if step_max > 1.0:
                gdm_orb_step *= 1.0/step_max
                step_max = numpy.max(numpy.abs(gdm_orb_step))

            g_pre  = numpy.dot(mf._grad_list[-1].T, gdm_orb_step)
            alpha2 = numpy.dot(gdm_orb_step.T, gdm_orb_step)
            alpha     = numpy.sqrt(alpha2)
            g_pre     = g_pre/alpha
            print("g_pre = ", g_pre)
            print("gdm_orb_step = \n", gdm_orb_step)
            print("grad         = \n", mf._grad_list[-1])
            alpha_pre = 0.0
            ene_pre   = ene_cur

            iter_line_search  = 0
            num_subspace_vec += 1
        else:
            log.debug("Line search!")
            mo_coeff = mf.mo_coeff
            print("ene_pre = %f, alpha_pre = %f, g_pre = %f"%(ene_pre, alpha_pre, g_pre))
            print("ene_cur = %f, alpha_cur = %f, g_cur = %f"%(ene_cur, alpha_cur, g_cur))
            alpha_cur = line_search_obj.next_step(ene_pre, alpha_pre, g_pre,
                                                  ene_cur, alpha_cur, g_cur,
                                                  is_wolfe1, is_wolfe2)
            print("ene_pre = %f, alpha_pre = %f, g_pre = %f"%(ene_pre, alpha_pre, g_pre))
            print("ene_cur = %f, alpha_cur = %f, g_cur = %f"%(ene_cur, alpha_cur, g_cur))
            print("alpha_cur = %f"%alpha_cur)
            print("alpha = %f"%alpha)
            print("gdm_orb_step = \n", gdm_orb_step)
            gdm_orb_step *= alpha_cur/alpha
            print("gdm_orb_step = \n", gdm_orb_step)
            step_max      = numpy.max(numpy.abs(gdm_orb_step))
            iter_line_search += 1

            if step_max > 1.0:
                gdm_orb_step *= 1.0/step_max
                step_max      = numpy.max(numpy.abs(gdm_orb_step))

        grad_diag_sqrt_h_inv  = grad * diag_sqrt_h_inv
        err        = numpy.max(numpy.abs(grad_diag_sqrt_h_inv))
        scf_conv   = (err < conv_tol)

        if mf._step_list is None:
            mf._step_list = []
            mf._step_list.append(gdm_orb_step)
        else:
            num_step = len(mf._step_list)
            if (num_subspace_vec-1 == num_step):
                mf._step_list.append(gdm_orb_step)
            elif num_subspace_vec-1 < num_step:
                mf._step_list[num_subspace_vec-1] = gdm_orb_step
            else:
                RuntimeError("GDM dimensioning error!")


        is_first_step = False

        pre_mo_coeff = cur_mo_coeff
        cur_mo_coeff = mf.update_mo_coeff(gdm_orb_step, pre_mo_coeff, mo_occ)

        ene_pre       = ene_cur
        dm_last       = dm
        dm   = mf.make_rdm1(cur_mo_coeff, mo_occ)
        dm   = lib.tag_array(dm, mo_coeff=cur_mo_coeff, mo_occ=mo_occ)
        print("dm = ")
        dump_rec(stdout, dm/2)
        iter_gdm += 1

    if scf_conv and conv_check:
        # An extra diagonalization, to remove level shift
        #fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm, dm_last = mf.make_rdm1(mo_coeff, mo_occ), dm
        dm  = lib.tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
        vhf = mf.get_veff(mol, dm, dm_last, vhf)
        e_tot, last_hf_e = mf.energy_tot(dm, h1e, vhf), e_tot

        fock = mf.get_fock(h1e, s1e, vhf, dm)
        norm_gorb = numpy.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
        if not TIGHT_GRAD_CONV_TOL:
            norm_gorb = norm_gorb / numpy.sqrt(norm_gorb.size)
        norm_ddm = numpy.linalg.norm(dm-dm_last)

        conv_tol      = conv_tol * 10
        conv_tol_grad = conv_tol_grad * 3
        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot-last_hf_e) < conv_tol or norm_gorb < conv_tol_grad:
            scf_conv = True
        logger.info(mf, 'Extra cycle  E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g',
                    e_tot, e_tot-last_hf_e, norm_gorb, norm_ddm)
        if dump_chk:
            mf.dump_chk(locals())
    logger.timer(mf, 'scf_cycle', *cput0)
    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ


# returns an exact gradient and approximate hessian
def get_orb_grad_hess_rhf(mf, mo_coeff, coo, cvv, mo_occ, fock_ao=None):
    if fock_ao is None:
        dm      = mf.make_rdm1(mo_coeff, mo_occ)
        fock_ao = mf.get_fock(dm=dm)

    occidx = numpy.where(mo_occ==2)[0]
    viridx = numpy.where(mo_occ==0)[0]
    print("fock_ao")
    dump_rec(stdout, fock_ao)
    f1   = reduce(numpy.dot, (mo_coeff.conj().T, fock_ao, mo_coeff))
    print("f1")
    dump_rec(stdout, f1)
    grad = f1[viridx[:,None],occidx] * 2

    mo_coeff_cano = numpy.empty_like(mo_coeff)
    orbo          = mo_coeff[:,occidx]
    orbv          = mo_coeff[:,viridx]
    mo_coeff_cano[:,occidx] = numpy.dot(orbo, coo)
    mo_coeff_cano[:,viridx] = numpy.dot(orbv, cvv)

    f1_cano   = reduce(numpy.dot, (mo_coeff_cano.conj().T, fock_ao, mo_coeff_cano))
    print("f1_cano")
    dump_rec(stdout, f1_cano)
    foo       = f1_cano[occidx[:,None],occidx]
    fvv       = f1_cano[viridx[:,None],viridx]

    h_diag = (fvv.diagonal().real[:,None] - foo.diagonal().real) * 2
    h_diag_min = 1.0/16.0
    abs_h_diag = numpy.abs(h_diag.T.reshape(-1))
    abs_h_diag[abs_h_diag<h_diag_min] = h_diag_min

    tmp_grad = grad.T.reshape(-1)
    return numpy.concatenate((tmp_grad, tmp_grad)), numpy.concatenate((abs_h_diag, abs_h_diag))

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
    e, cano_trans_orb_oo    = scipy.linalg.eigh(foo)
    mo_coeff_cano[:,occidx] = numpy.dot(orbo, cano_trans_orb_oo)
    mo_energy[occidx]       = e

    orbv   = mo_coeff[:,viridx]
    fvv    = reduce(numpy.dot, (orbv.conj().T, fock_ao, orbv))
    e, cano_trans_orb_vv    = scipy.linalg.eigh(fvv)
    mo_coeff_cano[:,viridx] = numpy.dot(orbv, cano_trans_orb_vv)
    mo_energy[viridx]       = e
    return cano_trans_orb_oo, cano_trans_orb_vv, mo_coeff_cano, mo_energy

def update_grad_list_rhf(mf, coo, cvv, grad_list):
    nocc = coo.shape[0]
    nvir = cvv.shape[0]
    tmp_grad_list = []
    for grad in grad_list:
        tmp_grad = grad.reshape(2, nocc, nvir)
        new_grad = numpy.einsum("ij,sia,ab->sjb", coo, tmp_grad, cvv)
        tmp_grad_list.append(new_grad.reshape(-1))
    return tmp_grad_list

def update_mo_coeff_rhf(mf, gdm_orb_step, mo_coeff, mo_occ):
    nmo     = len(mo_occ)
    occidxa = mo_occ==2
    occidxb = mo_occ==2
    viridxa = ~occidxa
    viridxb = ~occidxb
    idx     = (viridxa[:,None] & occidxa) | (viridxb[:,None] & occidxb)
    nocc = numpy.count_nonzero(mo_occ)
    nvir = nmo - nocc
    tmp_orb_step = gdm_orb_step.reshape(2,nocc,nvir)[0].T.reshape(-1)
    print("mo_coeff = ")
    dump_rec(stdout, mo_coeff)
    dtheta_vo      = numpy.zeros((nmo,nmo), dtype=gdm_orb_step.dtype)
    dtheta_vo[idx] = tmp_orb_step
    theta          = dtheta_vo - dtheta_vo.T
    print("theta = ")
    dump_rec(stdout, theta)
    u              = expm(theta)
    print("u = ")
    dump_rec(stdout, u)
    if mo_coeff is not None:
        u = numpy.dot(mo_coeff, u)
    tmp_mo_coeff = u
    print("tmp_mo_coeff = ")
    dump_rec(stdout, tmp_mo_coeff)
    return tmp_mo_coeff

class GDMOptimizer(object):
    def __init__(self, mf):
        self.__dict__.update(mf.__dict__)
        self._scf = mf

        self.max_cycle_gdm = MAX_CYCLE_GDM
        self.max_cycle_ls  = MAX_CYCLE_LS

        self._grad_list = None
        self._step_list = None

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        self._scf.dump_flags(verbose)
        log.info('******** %s GDM solver flags ********', self._scf.__class__)
        return self

    def proj_ew_subspace(self, diag_sqrt_h_inv):
        num_subspace_vec = len(self._grad_list)
        cur_norm = numpy.linalg.norm(self._grad_list[-1])
        for i in range(num_subspace_vec):
            igrad    = num_subspace_vec - i - 1
            grad     = self._grad_list[igrad]
            tmp_vec  = grad * diag_sqrt_h_inv
            tmp_norm = numpy.linalg.norm(tmp_vec)
            if tmp_norm > cur_norm*1e10 or (igrad==1 and num_subspace_vec==MAX_SUBSPACE_SIZE):
                del self._grad_list[0:igrad]
                del self._step_list[0:igrad]
                num_subspace_vec = num_subspace_vec - i
                break
        
        diag_sqrt_h    = 1.0/diag_sqrt_h_inv
        ew_subspace_1  = [grad*diag_sqrt_h_inv for grad in self._grad_list]
        if self._step_list is not None:
            ew_subspace_2 = [step*diag_sqrt_h for step in self._step_list]
            ew_subspace   = numpy.asarray(ew_subspace_1 + ew_subspace_2).T
        else:
            ew_subspace   = numpy.asarray(ew_subspace_1).T
        
        
        ew_subspace_t_dot_ew_subspace = dot(ew_subspace.T, ew_subspace)
        tmp_ew_subspace               = numpy.copy(ew_subspace)
        
        dim_subspace = ew_subspace_t_dot_ew_subspace.shape[0]
        tmp_diag     = numpy.diag(ew_subspace_t_dot_ew_subspace)
        for i in range(dim_subspace):
            for j in range(i+1,dim_subspace):
                tmp = ew_subspace_t_dot_ew_subspace[i,j]/numpy.sqrt(tmp_diag[i]*tmp_diag[j])
                ew_subspace_t_dot_ew_subspace[i,j] = tmp
                ew_subspace_t_dot_ew_subspace[j,i] = tmp
            ew_subspace_t_dot_ew_subspace[i,i] = 1.0

        eig_vals, eig_vecs = numpy.linalg.eigh(ew_subspace_t_dot_ew_subspace)
        sort_idx           = eig_vals.argsort()
        sort_eig_vals      = numpy.abs(eig_vals[sort_idx])
        tmp_eig_vals       = sort_eig_vals/sort_eig_vals[-1]
        sort_eig_vecs      = eig_vecs[:, sort_idx]
        
        tmp_idx            = tmp_eig_vals>1e-8
        tmp_vals           = sort_eig_vals[tmp_idx]
        inv_sqrt_vals      = 1/numpy.sqrt(tmp_vals)
        tmp_vecs           = sort_eig_vecs[:,tmp_idx]
        
        vecs_dot_inv_sqrt_vals = dot(tmp_vecs, numpy.diag(inv_sqrt_vals))
        for i in range(dim_subspace):
            tmp_norm             = numpy.linalg.norm(tmp_ew_subspace[:,i])
            tmp_ew_subspace[:,i] = tmp_ew_subspace[:,i]/tmp_norm
        
        tmp_mat = dot(tmp_ew_subspace, vecs_dot_inv_sqrt_vals)
        for i in range(tmp_vals.size):
            tmp_norm = numpy.linalg.norm(tmp_mat[:,i])
            tmp_mat[:,i] = tmp_mat[:,i]/tmp_norm
        subspace = dot(tmp_mat.T, ew_subspace)
        return subspace, tmp_mat

    def get_hess_inv(self, subspace):
        dim_subspace, tmp_num = subspace.shape
        num_subspace_vec = (tmp_num+1)//2
        gmat = subspace[:,:num_subspace_vec]
        smat = subspace[:,num_subspace_vec:(2*num_subspace_vec)]

        hess_inv = numpy.eye(dim_subspace)
        for i_subspace_vec in range(1,num_subspace_vec):
            g_new = gmat[:,i_subspace_vec]
            g_old = gmat[:,i_subspace_vec-1]
            
            yk    = (g_new - g_old).reshape(dim_subspace,1)
            sk    = (smat[:,i_subspace_vec-1]).reshape(dim_subspace,1)
            
            yk_dot_sk = dot(yk.T,sk)
            sk_dot_sk = dot(sk.T,sk)
            yk_dot_yk = dot(yk.T,yk)
            inv_hess_thresh = numpy.sqrt(sk_dot_sk) * numpy.sqrt(yk_dot_yk)
            
            if numpy.abs(yk_dot_sk) >= 1e-8 * inv_hess_thresh:
                u = numpy.eye(dim_subspace) - dot(sk, yk.T)/yk_dot_sk
                hess_inv = reduce(dot, [u, hess_inv, u.T]) + dot(sk,sk.T)/yk_dot_sk
            
        eig_vals, eig_vecs = numpy.linalg.eigh(hess_inv)
        abs_eig_vals       = numpy.abs(eig_vals)
        hess_inv           = reduce(dot, [eig_vecs, numpy.diag(abs_eig_vals), eig_vecs.T])
        return hess_inv
    
    def kernel(self):
        pass

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
            __doc__ = mf_doc
            get_orb_grad_hess = get_orb_grad_hess_rhf
            get_canonicalize  = get_canonicalize_rhf
            update_grad_list  = update_grad_list_rhf
            update_mo_coeff   = update_mo_coeff_rhf
        return SecondOrderRHF(mf)
