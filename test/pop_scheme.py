import numpy
from pyscf import gto, scf, dft

from pyscf.tools.dump_mat import dump_rec
from sys import stdout

class PopulationScheme(object):
    def __init__(self, mf, frg_list, do_spin_pop=False):
        assert isinstance(mf, scf.hf.SCF)
        assert isinstance(frg_list,      list)
        assert isinstance(frg_list[0],   list)
        assert isinstance(frg_list[0][0], int)

        self._scf               = mf
        self._mol               = mf.mol
        self._frg_list          = frg_list
        self.do_spin_pop        = do_spin_pop

        natm           = self._mol.natm
        natm_tmp       = 0
        atom_list_tmp  = []
        is_uniq_atm    = True
        is_frgm_in_mol = True

        for atm_list in frg_list:
            natm_tmp += len(atm_list)
            for atm_idx in atm_list:
                isinstance(atm_idx, int)
                is_uniq_atm    = not atm_idx in atom_list_tmp
                is_frgm_in_mol = 0 <= atm_idx < natm
                atom_list_tmp.append(atm_idx)
        
        is_frgm_in_mol = natm_tmp == natm
        assert is_frgm_in_mol
        assert is_uniq_atm

        self.frg_num    = len(frg_list)
        self.atm_num    = natm
        self.ao_num     = self._mol.nao

        if not self._scf.converged:
            mf.kernel()

        self.weight_matrices = None

    def get_frg_in_mol(self):
        return self._frg_list

    def get_atm_in_frg(self, ifrg):
        nfrg = self.frg_num
        assert 0 <= ifrg < nfrg and isinstance(ifrg, int)
        return self._frg_list[ifrg]

    def get_ao_in_atm(self, iatm):
        natm = self.atm_num
        assert 0 <= iatm < natm and isinstance(iatm, int)

        ao_labels = self._mol.ao_labels(fmt=None)
        ao_list  = []
        for i,s in enumerate(ao_labels):
            if s[0] == iatm:
                ao_list.append(i)
        return ao_list

    def get_ao_in_frg(self, ifrg):
        atm_list = self.get_atm_in_frg(ifrg)
        ao_in_frg = []
        for iatm in atm_list:
            ao_list    = self.get_ao_in_atm(iatm)
            ao_in_frg += ao_list
        return ao_in_frg

    def get_pop(self, density_matrix=None, weight_matrices=None,
                      do_spin_pop=None):
        if weight_matrices is None:
            weight_matrices = self.make_weight_matrices()
        if density_matrix  is None:
            density_matrix  = self._scf.make_rdm1()
        if do_spin_pop     is None:
            do_spin_pop     = self.do_spin_pop

        nfrg, nao = self.frg_num, self.ao_num
        assert weight_matrices.shape == (nfrg, nao, nao)

        if do_spin_pop:
            assert isinstance(self._scf, scf.uhf.UHF)
            assert density_matrix.shape == (2, nao, nao)
            dm  = density_matrix
            pop = numpy.einsum("fij,sji->fs", weight_matrices, dm)
            pop = pop.reshape(nfrg,2)
        else:
            if isinstance(self._scf, scf.uhf.UHF):
                assert density_matrix.shape == (2, nao, nao)
                dm = density_matrix[0] + density_matrix[1]
            else:
                assert density_matrix.shape == (nao, nao)
                dm = density_matrix

            pop = numpy.einsum("fij,ji->f", weight_matrices, dm)
            pop = pop.reshape(nfrg,1)
        
        return pop

    def make_frg_weight_matrix(self, ifrg):
        pass

    def make_weight_matrices(self):
        pass


class MullikenPopulation(PopulationScheme):
    def make_frg_weight_matrix(self, ifrg):
        nao = self.ao_num
        ao_in_frg  = self.get_ao_in_frg(ifrg)
        ovlp_ao    = self._scf.get_ovlp()
        for mu in range(nao):
            for nu in range(nao):
                if mu not in ao_in_frg and nu not in ao_in_frg:
                    ovlp_ao[mu,nu] = 0.0
                elif mu != nu:
                    ovlp_ao[mu,nu] = 0.5*ovlp_ao[mu,nu]
        return ovlp_ao

    def make_weight_matrices(self):
        nfrg = self.frg_num
        weight_matrices = [self.make_frg_weight_matrix(ifrg) for ifrg in range(nfrg)]
        return numpy.asarray(weight_matrices)


# nfrg = self.frg_num
