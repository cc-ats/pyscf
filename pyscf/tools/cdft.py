import numpy
from pyscf import gto, scf, dft, lib

from frag_pop import FragmentPopulation
from pyscf.tools.dump_mat import dump_rec
from sys import stdout

class Constraints(object):
    def __init__(self, frag_pop, nelec_required_list):
        self._scf                 = frag_pop._scf
        self._mol                 = frag_pop._mol
        self._frag_pop            = frag_pop
        self._nelec_required_list = numpy.asarray(nelec_required_list, dtype=float)

        self.do_spin_pop          = frag_pop.do_spin_pop

        assert isinstance(frag_pop,       FragmentPopulation)
        assert len(nelec_required_list) == frag_pop.frg_num
        self.constraints_num =  frag_pop.frg_num

        if self.do_spin_pop:
            assert self._nelec_required_list.shape == (self.constraints_num, 2)
        else:
            assert self._nelec_required_list.shape == (self.constraints_num, 1)

        assert numpy.sum(self._nelec_required_list) == self._mol.nelectron

    def get_pop_shape(self):
        return self._nelec_required_list.shape

    def get_pop_minus_nelec_required(self, density_matrix=None, weight_matrices=None,
                                           do_spin_pop=None):
        pop = self._frag_pop.get_pop(density_matrix=density_matrix, weight_matrices=weight_matrices, do_spin_pop=do_spin_pop)
        return pop-self._nelec_required_list

    def make_weight_matrices(self, method="Mulliken"):
        return self._frag_pop.make_weight_matrices(method=method)

    def make_cdft_fock_add(self, lam_vals, weight_matrices, do_spin_pop=None):
        if do_spin_pop     is None:
            do_spin_pop     = self.do_spin_pop
        lam_val_list = numpy.asarray(lam_vals)
        assert lam_val_list.shape == self._nelec_required_list.shape
        if self.do_spin_pop:
            assert self._nelec_required_list.shape == (self.constraints_num, 2)
        else:
            assert self._nelec_required_list.shape == (self.constraints_num, 1)

        fock_add_ao = numpy.einsum("fs,fij->sij", lam_vals, weight_matrices)
        if do_spin_pop:
            tmp_fock_ao = [fock_add_ao[0], fock_add_ao[1]]
            return numpy.asarray(tmp_fock_ao)
        else:
            if isinstance(self._scf, scf.uhf.UHF):
                tmp_fock_ao = [fock_add_ao[0], fock_add_ao[0]]
                return numpy.asarray(tmp_fock_ao)
            else:
                tmp_fock_ao = fock_add_ao[0]
                return numpy.asarray(tmp_fock_ao)

    def make_cdft_grad(self, density_matrix, weight_matrices, do_spin_pop=None):
        return self.get_pop_minus_nelec_required(self, density_matrix=density_matrix, weight_matrices=weight_matrices, do_spin_pop=do_spin_pop)

    def make_cdft_hess(self, lam_vals, weight_matrices, do_spin_pop=None):
        mo_coeff  = self._scf.mo_coeff
        mo_occ    = self._scf.mo_occ
        mo_energy = self._scf.mo_energy

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
            e_vo_a = mo_energy[1][occidxb][:,None] - mo_energy[1][viridxb]
            e_vo   = numpy.asarray([e_vo_a, e_vo_b])

            wvo_a = numpy.einsum("am,mn,in->sai", orbva, wao, orboa)
            wvo_b = numpy.einsum("am,mn,in->sai", orbvb, wao, orbob)
            wvo   = numpy.asarray([wvo_a, wvo_b])

            if do_spin_pop:
                hess = numpy.einsum("smai,snai,sai->smn", wvo, wvo, 1./e_vo)
            else:
                hess = numpy.einsum("smai,snai,sai->mn", wvo, wvo, 1./e_vo)

        elif mo_coeff.ndim == 2:
            occidx = numpy.where(mo_occ==2)[0]
            viridx = numpy.where(mo_occ==0)[0]
            nocc = len(occidx)
            nvir = len(viridx)
            orbv = mo_coeff[:,viridx]
            orbo = mo_coeff[:,occidx]
            e_vo = mo_energy[viridx][:,None] - mo_energy[occidx]

            wvo  = numpy.einsum("am,mn,in->ai", orbv, wao, orbo)
            hess = numpy.einsum("mai,nai,ai->mn", wvo, wvo, 1./e_vo) 

        return hess

def cdft_inner_cycle(mf, frg_list, nelec_required_list, init_lam_list,
                         dm0=None, old_get_fock=None, old_energy_elec=None,
                         diis_class=None, diis_space=8, diis_start_cycle=8,
                         frag_pop="mulliken", do_spin_pop=False,
                         tol=1e-8,  maxiter=200, verbose=0):
    mf.verbose   = verbose
    mf.max_cycle = maxiter
    mf.conv_tol  = tol

    if dm0 is None:
        dm0 = mf.make_rdm1()
    if old_get_fock is None:
        old_get_fock       = mf.get_fock
    if old_energy_elec is None:
        old_energy_elec    = mf.energy_elec

    if frag_pop is "mulliken":
        pop_method = FrgMullikenPopulation(mf, frg_list, do_spin_pop=do_spin_pop)
    else:
        RuntimeError("Don't support the population scheme!")

    init_lam_vals   = numpy.asarray(init_lam_list)
    con_vals        = numpy.asarray(nelec_required_list)
    assert init_lam_vals.shape == con_vals.shape

    constraints     = Constraints(pop_method, nelec_required_list)
    weight_matrices = constraints.make_weight_matrices()

    if diis_class is None:
        cdft_diis       = scf.diis.SCF_DIIS(mf, mf.diis_file)
        cdft_diis.space = diis_space
    elif issubclass(diis_class, lib.diis.DIIS):
        cdft_diis       = diis_class(mf, mf.diis_file)
        cdft_diis.space = diis_space

    fock_add = constraints.make_cdft_fock_add(init_lam_vals, weight_matrices=weight_matrices, do_spin_pop=do_spin_pop)

    def get_fock(h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, mf_diis=None):
        if cycle < 0:
            fock_0 = old_get_fock(h1e=h1e, s1e=s1e, vhf=vhf, dm=dm,
                                  cycle=-1, diis=None)
            fock     = fock_0 + fock_add
            return fock
        else:
            fock_0 = old_get_fock(h1e=h1e, s1e=s1e, vhf=vhf, dm=dm,
                                  cycle=-1, diis=None)
            fock     = fock_0 + fock_add
            if s1e is None: s1e = mf.get_ovlp()
            if cycle >= diis_start_cycle:
                fock = cdft_diis.update(s1e, dm, fock, mf, h1e, vhf)
            return fock

    mf.get_fock    = get_fock
    mf.kernel(dm0)

    dm1   = mf.make_rdm1()
    pop_minus_nelec_required = constraints.get_pop_minus_nelec_required(
        density_matrix=dm1, weight_matrices=weight_matrices,do_spin_pop=do_spin_pop
    )
    return pop_minus_nelec_required

def cdft_grad()


def cdft(mf, frg_list, nelec_required_list, init_lam_list,
                         dm0=None, old_get_fock=None, old_energy_elec=None,
                         diis_class=None, diis_space=8, diis_start_cycle=8,
                         frag_pop="mulliken", do_spin_pop=False,
                         tol=1e-8,  maxiter=200, verbose=0):
    pass