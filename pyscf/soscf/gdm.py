import sys
from sys import stdout
import time
from functools import reduce

import numpy
import scipy
from scipy.linalg import expm
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

def _effective_svd(a, tol=SVD_TOL):
    w = svd(a)[1]
    return w[(tol<w) & (w<1-tol)]

def expmat(a):
    return expm(a)

def kernel(mf, mo_coeff=None, mo_occ=None, dm=None,
           conv_tol=1e-10, conv_tol_grad=None, max_cycle=50, dump_chk=True,
           callback=None, verbose=logger.NOTE):
    cput0 = (time.clock(), time.time())
    log = logger.new_logger(mf, verbose)
    mol = mf._scf.mol
    if mol != mf.mol:
        logger.warn(mf, 'dual-basis SOSCF is an experimental feature. It is '
                    'still in testing.')

    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(conv_tol)
        log.info('Set conv_tol_grad to %g', conv_tol_grad)

# call mf._scf.get_hcore, mf._scf.get_ovlp because they might be overloaded
    h1e = mf._scf.get_hcore(mol)
    s1e = mf._scf.get_ovlp(mol)

    if mo_coeff is not None and mo_occ is not None:
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        # call mf._scf.get_veff, to avoid "newton().density_fit()" polluting get_veff
        vhf = mf._scf.get_veff(mol, dm)
        fock = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
        mo_energy, mo_tmp = mf.eig(fock, s1e)
        mf.get_occ(mo_energy, mo_tmp)
        mo_tmp = None

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
        # If different direct_scf_cutoff is assigned to newton_ah mf.opt
        # object, mf.opt should be different to mf._scf.opt
        #mf.opt = mf._scf.opt

    scf_conv = False
    while not scf_conv:
        pass

    for imacro in range(max_cycle):
        u, g_orb, kfcount, jkcount, dm_last, vhf = \
                rotaiter.send((mo_coeff, mo_occ, dm, vhf, e_tot))
        kftot += kfcount + 1
        jktot += jkcount + 1

        last_hf_e = e_tot
        norm_gorb = numpy.linalg.norm(g_orb)
        mo_coeff = mf.rotate_mo(mo_coeff, u, log)
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        vhf = mf._scf.get_veff(mol, dm, dm_last=dm_last, vhf_last=vhf)
        fock = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
# NOTE: DO NOT change the initial guess mo_occ, mo_coeff
        if mf.verbose >= logger.DEBUG:
            mo_energy, mo_tmp = mf.eig(fock, s1e)
            mf.get_occ(mo_energy, mo_tmp)
# call mf._scf.energy_tot for dft, because the (dft).get_veff step saved _exc in mf._scf
        e_tot = mf._scf.energy_tot(dm, h1e, vhf)

        log.info('macro= %d  E= %.15g  delta_E= %g  |g|= %g  %d KF %d JK',
                 imacro, e_tot, e_tot-last_hf_e, norm_gorb,
                 kfcount+1, jkcount)
        cput1 = log.timer('cycle= %d'%(imacro+1), *cput1)

        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot-last_hf_e) < conv_tol and norm_gorb < conv_tol_grad:
            scf_conv = True

        if dump_chk:
            mf.dump_chk(locals())
    
    mo_energy, mo_coeff1 = mf._scf.canonicalize(mo_coeff, mo_occ, fock)
    if mf.canonicalization:
        log.info('Canonicalize SCF orbitals')
        mo_coeff = mo_coeff1
        if dump_chk:
            mf.dump_chk(locals())
    log.info('macro X = %d  E=%.15g  |g|= %g  total %d KF %d JK',
             imacro+1, e_tot, norm_gorb, kftot+1, jktot+1)
    if (numpy.any(mo_occ==0) and
        mo_energy[mo_occ>0].max() > mo_energy[mo_occ==0].min()):
        log.warn('HOMO %s > LUMO %s was found in the canonicalized orbitals.',
                 mo_energy[mo_occ>0].max(), mo_energy[mo_occ==0].min())
    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

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
    '''Canonicalization diagonalizes the Fock matrix within occupied, open,
    virtual subspaces separatedly (without change occupancy).
    '''
    if fock_ao is None:
        dm      = mf.make_rdm1(mo_coeff, mo_occ)
        fock_ao = mf.get_fock(dm=dm)

    occidx = mo_occ == 2
    viridx = mo_occ == 0

    mo   = numpy.empty_like(mo_coeff)
    mo_e = numpy.empty(mo_occ.size)

    orbo   = mo_coeff[:,occidx]
    foo    = reduce(numpy.dot, (orbo.conj().T, fock_ao, orbo))
    e, cano_trans_orb_oo = scipy.linalg.eig(foo)
    mo[:,occidx] = numpy.dot(orbo, cano_trans_orb_oo)

    orbv   = mo_coeff[:,viridx]
    fvv    = reduce(numpy.dot, (orbv.conj().T, fock_ao, orbv))
    e, cano_trans_orb_vv = scipy.linalg.eig(fvv)
    mo[:,viridx] = numpy.dot(orbv, cano_trans_orb_vv)

    return cano_trans_orb_oo, cano_trans_orb_vv, mo


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

    gdm_start_tol   = getattr(__config__, 'soscf_gdm_start_tol',  1e9)
    gdm_start_cycle = getattr(__config__, 'soscf_gdm_start_cycle',  1)
    gdm_level_shift = getattr(__config__, 'soscf_gdm_level_shift',0.0)
    gdm_conv_tol    = getattr(__config__, 'soscf_gdm_conv_tol', 1e-12)
    gdm_lindep      = getattr(__config__, 'soscf_gdm_lindep',   1e-14)
    gdm_max_cycle   = getattr(__config__, 'soscf_gdm_max_cycle',   40)

    max_cycle_ls     = getattr(__config__, 'soscf_gdm_max_cycle_inner'   , 12)
    max_stepsize_ls  = getattr(__config__, 'soscf_gdm_max_stepsize',      .05)
    

    def __init__(self, mf):
        self.__dict__.update(mf.__dict__)
        self._scf = mf

        self._keys.update((   'max_cycle_ls',
                           'max_stepsize_ls',
                             'gdm_start_tol',
                           'gdm_start_cycle',
                           'gdm_level_shift',
                              'gdm_conv_tol',
                                'gdm_lindep',
                             'gdm_max_cycle'))

        self.effective_thresh = DEF_EFF_THRSH

        self.ene_pre = 0.0
        self.ene_cur = 0.0

        self.x_cur = 0.0
        self.g_cur = 0.0
        self.g_ref = 0.0
        self.x_ref = 0.0
        self.ref_len = 0.0
        self.r_trust = 0.0
        self.de_model = 0.0
        self.step_max = 0.0

        self.rms = 0.0
        self.error = 0.0

        self.is_first_step    = True
        self.nssv             = 0
        self.iter_line_search = 0

        # Line search wolfe1 condition 1
        self.is_wolfe1 = True
        # Line search wolfe1 condition 1
        self.is_wolfe2 = True

        self.gdm_step_vec = None
        self.step_list    = None
        self.grad_list    = None

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        self._scf.dump_flags(verbose)
        log.info('******** %s GDM solver flags ********', self._scf.__class__)
        return self

    def build(self, mol=None):
        if mol is None: mol = self.mol
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self._scf.build(mol)
        self.opt = None
        self._eri = None
        return self

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        return self._scf.reset(mol)

    def update_rotate_matrix(self, dx, mo_occ, u0=1, mo_coeff=None):
        dr = hf.unpack_uniq_var(dx, mo_occ)
        return numpy.dot(u0, expmat(dr))

    def rotate_mo(self, mo_coeff, u, log=None):
        mo = numpy.dot(mo_coeff, u)

        if isinstance(log, logger.Logger) and log.verbose >= logger.DEBUG:
            idx = self.mo_occ > 0
            s = reduce(numpy.dot, (mo[:,idx].conj().T, self._scf.get_ovlp(),
                                   self.mo_coeff[:,idx]))
            log.debug('Overlap to initial guess, SVD = %s',
                      _effective_svd(s, 1e-5))
            log.debug('Overlap to last step, SVD = %s',
                      _effective_svd(u[idx][:,idx], 1e-5))
        return mo

    def update_orb(self, orb_step, mo_coeff=None, mo_occ=None, dm0=None,
                         s1e=None, hie=None, vhf0=None):
        if s1e is None:
            s1e = self._scf.get_ovlp()
        if h1e is None:
            h1e = self._scf.get_hcore()

        u = self.update_rotate_matrix(orb_step, mo_occ, mo_coeff=mo_coeff)
        if mo_coeff is not None:
            u = self.rotate_mo(mo_coeff, u)
        mo_coeff = u

        return mo_coeff

    def update_hess_inv(self): # BFGS update hess inv
        pass

    def save_orb(self, mo_coeff):
        self.mo_coeff = mo_coeff

    def next_gdm_step(self, ene_prev, mo_coeff, mo_occ, fock_ao, verbose=None, iter_gdm_step=None):
        if verbose is None:
            verbose = self.verbose

        log = logger.new_logger(self, verbose)
        log.note('\n')
        log.info('Geometric Direct Minimization')

        grad, diag_hess = self.get_orb_grad_hess(mo_coeff, mo_occ, fock_ao=fock_ao)
        prec  = 1.0/numpy.sqrt(ndiag_hess)

        self.ene_pre = self.e_tot
        self.ene_cur = self._scf.energy_tot()

        norm_gorb = numpy.linalg.norm(grad)
        norm_gorb = norm_gorb / numpy.sqrt(norm_gorb.size)

        if (norm_gorb < self.gdm_conv_tol * self.gdm_conv_tol):
            log.debug("---------- Successful GDM Step ----------")
            log.debug("Hopefully we are in the right direction.")
            self.e_tot = self.ene_cur
            return True

        if self.gdm_step_vec is None:
            self.gdm_step_vec = numpy.empty_like(grad)

        if self.grad_list is None:
            self.grad_list = []
            self.grad_list.append(grad)
        else:
            num_grad = len(self.grad_list)
            if (self.nssv == num_grad):
                self.grad_list.append(grad)
            elif (self.nssv < num_grad):
                self.grad_list[self.nssv] = grad
            else:
                RuntimeError("GDM dimensioning error!")

        if not self.is_first_step:
            diff_ene = self.ene_cur-self.ene_pre
            self.trust_radius = self.get_trust_radius(diff_ene)
            if -diff_ene >= 1e-4*self.g_ref:
                self.is_wolfe1 = True
            elif numpy.abs(diff_ene) <= self.effective_thresh:
                self.is_wolfe1 = True
            else:
                self.is_wolfe1 = False
            
            self.g_cur = numpy.dot(self.gdm_step_vec, grad)
            xx         = numpy.dot(self.gdm_step_vec, self.gdm_step_vec)
            self.x_cur = numpy.sqrt(xx)
            self.g_cur = self.g_cur/self.x_cur

            if self.g_cur >= CURV_CONDITION_SCALE * self.g_ref:
                self.is_wolfe2 = True
                if self.iter_line_search > 15:
                    self.is_wolfe2     = True
                    self.is_first_step = True

        if self.is_wolfe1 and self.is_wolfe2: 
            self.iter_line_search = 0
            self.save_orb()
            tmp_grad_list = numpy.einsum("ab,tai,ij->tbj", c_vv, self.grad_list, c_oo)
            self.grad_list = tmp_grad_list
            tmp_step_list = numpy.einsum("ab,tai,ij->tbj", c_vv, self.step_list, c_oo)
            self.step_list = tmp_step_list

            nssv += 1
            n_ss  = -1

            subspace_mat, ene_weight_subspace, proj_err = self.proj_ene_weight_subspace()

            self.eff_thresh = numpy.max(proj_err, self.eff_thresh)
            qn_step = self.get_qn_step(subspace_mat)

            self.gdm_step_vec = numpy.dot(ene_weight_subspace, qn_step)
        else:
            self.back_to_origin()
            line_search_iter += 1
            self.do_line_search()
            step *= alpha_cur/sqrt(xx)

            step_max = numpy.max(step)


        s3  = grad * prec
        err = numpy.max(numpy.abs(s3))
        self._step_list.append(step)
        self.update(step)

        self.is_first_step = False



        return 

    def next_line_search_step(self, iter_line_search):
        pass

    def next_dogleg_step(self):
        pass

    get_orb_grad_hess = get_orb_grad_hess_rhf

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

                if WITH_EX_EY_DEGENERACY:
                    mol = self._scf.mol
                    if mol.symmetry and mol.groupname in ('Dooh', 'Coov'):
                        orbsyma, orbsymb = uhf_symm.get_orbsym(mol, mo_coeff)
                        _force_Ex_Ey_degeneracy_(dr[0], orbsyma)
                        _force_Ex_Ey_degeneracy_(dr[1], orbsymb)

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
