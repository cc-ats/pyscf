import numpy
from pyscf import gto, scf, dft, lib

from pop_scheme import FrgPopulationScheme, FrgMullikenPopulation
from pyscf.tools.dump_mat import dump_rec
from sys import stdout

class Constraints(object):
    def __init__(self, pop_scheme, nelec_required_list):
        self._scf                 = pop_scheme._scf
        self._mol                 = pop_scheme._scf.mol
        self._pop_scheme          = pop_scheme
        self._nelec_required_list = numpy.asarray(nelec_required_list)

        self.do_spin_pop          = pop_scheme.do_spin_pop

        assert isinstance(pop_scheme,       FrgPopulationScheme)
        assert len(nelec_required_list) == pop_scheme.frg_num
        self.constraints_num            =  pop_scheme.frg_num

        if self.do_spin_pop:
            len_nelec_required = 2
            assert self._nelec_required_list.shape == (self.constraints_num, 2)
        else:
            len_nelec_required = 1
            assert self._nelec_required_list.shape == (self.constraints_num, 1)
        
        for nelec_required in nelec_required_list:
            assert isinstance(nelec_required, list)
            assert len(nelec_required) == len_nelec_required
            for num_e in nelec_required:
                assert isinstance(num_e, float)

    def get_pop_shape(self):
        return self._nelec_required_list.shape

    def get_pop_minus_nelec_required(self, density_matrix=None, weight_matrices=None,
                                           do_spin_pop=None):
        pop = self._pop_scheme.get_pop(density_matrix=density_matrix, weight_matrices=weight_matrices, do_spin_pop=do_spin_pop)
        return pop-self._nelec_required_list

    def make_weight_matrices(self):
        return self._pop_scheme.make_weight_matrices()

    def make_fock_add(self, dms, omega_vals, weight_matrices=None, do_spin_pop=None):
        if weight_matrices is None:
            weight_matrices = self.make_weight_matrices()
        if do_spin_pop     is None:
            do_spin_pop     = self.do_spin_pop
        
        assert omega_vals.shape == self._nelec_required_list.shape
        pop_minus_nelec_required = self.get_pop_minus_nelec_required(
            density_matrix=dms, weight_matrices=weight_matrices, do_spin_pop=do_spin_pop
            )
        fock_add_ao = numpy.einsum("fs,fs,fij->sij", omega_vals, pop_minus_nelec_required, weight_matrices)
        if do_spin_pop:
            tmp = [fock_add_ao[0], fock_add_ao[1]]
            return numpy.asarray(tmp)
        else:
            if isinstance(self._scf, scf.uhf.UHF):
                tmp = [fock_add_ao[0], fock_add_ao[0]]
                return numpy.asarray(tmp)
            else:
                tmp = fock_add_ao[0]
                return numpy.asarray(tmp)


def rdft(mf, frg_list,  nelec_required_list,  omega_list, dm0=None,
             pop_scheme="mulliken", do_spin_pop=False,
             tol=1e-8,  maxiter=200, 
             verbose=0, diis_pos='post',      diis_type=1):
    mf.verbose   = verbose
    mf.max_cycle = maxiter
    mf.conv_tol  = tol
    old_get_fock       = mf.get_fock
    old_energy_elec    = mf.energy_elec

    if pop_scheme is "mulliken":
        pop_method = FrgMullikenPopulation(mf, frg_list, do_spin_pop=do_spin_pop)
    else:
        RuntimeError("Don't support the population scheme!")

    omega_vals   = numpy.asarray(omega_list)
    con_vals     = numpy.asarray(nelec_required_list)
    assert omega_vals.shape == con_vals.shape

    constraints     = Constraints(pop_method, nelec_required_list)
    weight_matrices = constraints.make_weight_matrices()

    rdft_diis       = lib.diis.DIIS()
    rdft_diis.space = 15

    def get_fock(h1e, s1e, vhf, dm, cycle=0, mf_diis=None):
        fock_0 = old_get_fock(h1e, s1e, vhf, dm, cycle, None)
        if mf_diis is None:
            fock_add = constraints.make_fock_add(dm, omega_vals, weight_matrices=weight_matrices, do_spin_pop=do_spin_pop)
            return fock_0 + fock_add

        fock_0   = old_get_fock(h1e, s1e, vhf, dm, cycle, None)
        fock_add = constraints.make_fock_add(dm, omega_vals, weight_matrices=weight_matrices, do_spin_pop=do_spin_pop)
        fock     = fock_0 + fock_add #ZHC

        if cycle > 1:
            if diis_type == 1:
                fock = rdft_diis.update(fock_0, scf.diis.get_err_vec(s1e, dm, fock)) + fock_add
            elif diis_type == 2:
                # TO DO difference < threshold...
                fock = rdft_diis.update(fock)
            elif diis_type == 3:
                fock = rdft_diis.update(fock, scf.diis.get_err_vec(s1e, dm, fock))
            else:
                print("\nWARN: Unknow CDFT DIIS type, NO DIIS IS USED!!!\n")

        return fock

    def energy_elec(dm=None, h1e=None, vhf=None):
        e0    = old_energy_elec(dm,h1e,vhf)
        pop_minus_nelec_required = constraints.get_pop_minus_nelec_required(
            density_matrix=dm, weight_matrices=weight_matrices,do_spin_pop=do_spin_pop
        )
        pop_minus_nelec_required2 = pop_minus_nelec_required * pop_minus_nelec_required
        e_add = numpy.einsum("fs,fs->", pop_minus_nelec_required2, omega_vals)
        if verbose > 3:
            print("e0 = {: 12.6f}, e_add = {: 12.6f}".format(e0[0], e_add))
        return (e0[0]+e_add, e0[1])

    mf.get_fock    = get_fock
    mf.energy_elec = energy_elec
    mf.kernel(dm0)

    dm1   = mf.make_rdm1()
    e0    = old_energy_elec(dm1)
    pop_minus_nelec_required = constraints.get_pop_minus_nelec_required(
        density_matrix=dm1, weight_matrices=weight_matrices,do_spin_pop=do_spin_pop
    )
    pop_minus_nelec_required2 = pop_minus_nelec_required * pop_minus_nelec_required
    e_add = numpy.einsum("fs,fs->", pop_minus_nelec_required2, omega_vals)

    return e0[0], e_add, dm1