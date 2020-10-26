import time
import numpy
import pyscf
from pyscf     import lib
from pyscf.lib import logger

from pyscf import gto
from pyscf import scf
from pyscf import dft
from pyscf.dft import numint

from copy import deepcopy
from functools import reduce

from pyscf import __config__

MUTE_CHKFILE     = getattr(__config__, 'mute_chkfile',                False)
NELEC_ERROR_TOL  = getattr(__config__, 'dft_rks_prune_error_tol',      0.02)
SMALL_RHO_CUTOFF = getattr(__config__, 'dft_rks_RKS_small_rho_cutoff', 1e-7)

class FragmentPopulation(lib.StreamObject):
    def __init__(self, mf_list, verbose=3, do_spin_pop=False):
        assert len(mf_list) > 1
        assert isinstance(mf_list,          list)
        assert isinstance(mf_list[0], scf.hf.SCF)

        self._scf_class    = mf_list[0].__class__
        if hasattr(mf_list[0], "xc"):
            self._scf_xc       = mf_list[0].xc

        for frag_mf in mf_list:
            assert frag_mf.converged
            assert isinstance(frag_mf, self._scf_class)
            if hasattr(frag_mf, "xc"):
                assert hasattr(self, "_scf_xc")
                assert frag_mf.xc == self._scf_xc


        self.verbose        = verbose
        
        self.frag_list      = mf_list
        self._frag_mol_list = [frag_mf.mol for frag_mf in mf_list]

        self._mol           = reduce(gto.mole.conc_mol, self._frag_mol_list)
        self._scf           = self._scf_class(self._mol)
        self._scf.max_cycle = mf_list[0].max_cycle
        self._scf.conv_tol  = mf_list[0].conv_tol

        self.atm_num        = self._mol.natm
        self.ao_num         = self._mol.nao
        self.frag_num       = len(mf_list)

        temp_index             = list(range(self.atm_num))
        self._atm_in_frag_list = []
        count = 0
        for frag_mol in self._frag_mol_list:
            self._atm_in_frag_list.append(temp_index[count:count+frag_mol.natm])
            count = count + frag_mol.natm

        self._ao_in_atm_list  = []
        ao_labels = self._mol.ao_labels(fmt=None)
        for iatm in range(self.atm_num):
            tmp_list = []
            for i,s in enumerate(ao_labels):
                if s[0] == iatm:
                    tmp_list.append(i)
            self._ao_in_atm_list.append(tmp_list)

        self._ao_in_frag_list = []
        for atm_in_frag in self._atm_in_frag_list:
            tmp_list = []
            for atm in atm_in_frag:
                tmp_list += self._ao_in_atm_list[atm]
            self._ao_in_frag_list.append(tmp_list)
        
        self.do_spin_pop    = do_spin_pop
        self.weight_matrix  = None

    def build(self, dm0=None):
        self._scf.kernel(dm0=dm0)
        dm = self._scf.make_rdm1()
        self._grid = dft.gen_grid.Grids(self._mol)
        self._grid.build()
        self._grid.coords, self._grid.weights, self._grid.non0tab = self._prune_small_rho_grids(
        dm=dm, coords=self._grid.coords, weights=self._grid.weights
        )

        logger.info(self, '')
        if hasattr(self, '_scf_xc'):
            logger.info(self, 'Exchange correlation functional is %s', self._scf_xc)
        
        for imf, frag_mf in enumerate(self.frag_list):
            logger.info(self, 'The fragment %d energy is %f', imf, frag_mf.e_tot)

        logger.info(self, 'The total energy      is %f', self._scf.e_tot)

    def _prune_small_rho_grids(self, dm=None, coords=None, weights=None, non0tab=None):
        if dm is None:
            dm = self._scf.make_rdm1()
        if dm.ndim == 3:
            dm = dm[0] + dm[1]

        grid = self._grid
        mol  = self._mol
        size = weights.size

        if coords is None:
            coords = grid.coords
        if weights is None:
            weights = grid.weights
        if non0tab is None:
            non0tab = grid.non0tab
        ao_value = numint.eval_ao( mol,   coords, deriv=0)
        rho      = numint.eval_rho(mol, ao_value, dm, xctype='lda')

        n   = numpy.dot(rho, weights)
        if abs(n-mol.nelectron) < NELEC_ERROR_TOL*n:
            rho *= weights
            idx = abs(rho) > SMALL_RHO_CUTOFF / weights.size
            self._drop_idx = abs(rho) <= SMALL_RHO_CUTOFF / weights.size
            coords  = numpy.asarray(coords [idx], order='C')
            weights = numpy.asarray(weights[idx], order='C')
            non0tab = grid.make_mask(mol, coords)
        logger.info(self, 'Drop grid %d',
                        size - numpy.count_nonzero(idx))
        return coords, weights, non0tab

    def get_atm_in_frag(self, ifrag):
        nfrag = self.frag_num
        assert 0 <= ifrag < nfrag and isinstance(ifrag, int)
        return self._atm_in_frag_list[ifrag]

    def get_ao_in_atm(self, iatm):
        natm = self.atm_num
        assert 0 <= iatm < natm and isinstance(iatm, int)
        return self._ao_in_atm_list[iatm]

    def get_ao_in_frag(self, ifrag):
        nfrag = self.frag_num
        assert 0 <= ifrag < nfrag and isinstance(ifrag, int)
        return self._ao_in_frag_list[ifrag]

    def get_pop(self, weight_matrix, density_matrix=None, do_spin_pop=None):
        if density_matrix  is None:
            density_matrix  = self._scf.make_rdm1()
        if do_spin_pop     is None:
            do_spin_pop     = self.do_spin_pop

        nfrag, nao = self.frag_num, self.ao_num
        assert weight_matrix.shape == (nfrag, nao, nao)

        if do_spin_pop:
            assert isinstance(self._scf, scf.uhf.UHF)
            assert density_matrix.shape == (2, nao, nao)
            dm  = density_matrix
            pop = numpy.einsum("fij,sji->fs", weight_matrix, dm)
            pop = pop.reshape(nfrag,2)
        else:
            if isinstance(self._scf, scf.uhf.UHF):
                assert density_matrix.shape == (2, nao, nao)
                dm = density_matrix[0] + density_matrix[1]
            else:
                assert density_matrix.shape == (nao, nao)
                dm = density_matrix

            pop = numpy.einsum("fij,ji->f", weight_matrix, dm)
            pop = pop.reshape(nfrag,1)
        
        return pop

    def make_weight_matrix_ao(self, method="Mulliken"):
        nfrag = self.frag_num
        nao   = self.ao_num
        weight_matrix = numpy.zeros([nfrag, nao, nao])
        if method.lower() == "mulliken" or method.lower() == "mul":
            ovlp_ao         = self._scf.get_ovlp()
            for ifrag, ao_list in enumerate(self._ao_in_frag_list):
                for mu in range(0, nao):
                    for nu in range(mu, nao):
                        if mu not in ao_list and nu not in ao_list:
                            weight_matrix[ifrag, mu, nu] = 0.0
                            weight_matrix[ifrag, nu, mu] = 0.0
                        elif (mu in ao_list and nu not in ao_list) or (mu not in ao_list and nu in ao_list):
                            tmp_val = 0.5*ovlp_ao[mu, nu]
                            weight_matrix[ifrag, mu, nu] = tmp_val
                            weight_matrix[ifrag, nu, mu] = tmp_val
                        elif (mu in ao_list and nu in ao_list):
                            tmp_val = ovlp_ao[mu, nu]
                            weight_matrix[ifrag, mu, nu] = tmp_val
                            weight_matrix[ifrag, nu, mu] = tmp_val
        else:
            raise NotImplementedError()
        
        return weight_matrix

    def make_weight_matrix_mo(self, wao):
        mo_coeff = self._scf.mo_coeff
        if mo_coeff.ndim == 3:
            wmo = numpy.einsum("smp,mn,snq->spq", mo_coeff, wao, mo_coeff)
        elif mo_coeff.ndim == 2:
            wmo = numpy.einsum("mp,mn,nq->pq", mo_coeff, wao, mo_coeff)
        else:
            raise NotImplementedError("Wrong dimension!")
        return wmo

    def make_weight_matrix_vo(self, wao):
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
            wvo_a = numpy.einsum("ma,mn,ni->sai", orbva, wao, orboa)
            wvo_b = numpy.einsum("ma,mn,ni->sai", orbvb, wao, orbob)
            wvo = numpy.array([wvo_a, wvo_b])

        elif mo_coeff.ndim == 2:
            occidx = numpy.where(mo_occ==2)[0]
            viridx = numpy.where(mo_occ==0)[0]
            nocc = len(occidx)
            nvir = len(viridx)
            orbv = mo_coeff[:,viridx]
            orbo = mo_coeff[:,occidx]

            orboa = mo_coeff[0][:,occidxa]
            orbob = mo_coeff[1][:,occidxb]
            orbva = mo_coeff[0][:,viridxa]
            orbvb = mo_coeff[1][:,viridxb]
            wmo = numpy.einsum("mp,mn,nq->pq", mo_coeff, wao, mo_coeff)
        else:
            raise NotImplementedError("Wrong dimension!")
        return wmo


if __name__ == "__main__":
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
    w  = fp.make_weight_matrix_ao(method="mulliken")
    print(fp.get_pop(w))

    frag1 = scf.UHF(mol1)
    frag1.kernel()

    frag2 = scf.UHF(mol2)
    frag2.kernel()

    frag_list = [frag1, frag2]
    fp = FragmentPopulation(frag_list, verbose=0)
    fp.build()
    w  = fp.make_weight_matrix_ao(method="mulliken")
    print(fp.get_pop(w))
    print(fp.get_pop(w, do_spin_pop=True))

    frag1 = dft.RKS(mol1)
    frag1.xc = 'pbe'
    frag1.kernel()

    frag2 = dft.RKS(mol2)
    frag2.xc = 'pbe'
    frag2.kernel()

    frag_list = [frag1, frag2]
    fp = FragmentPopulation(frag_list, verbose=0)
    fp.build()
    w  = fp.make_weight_matrix_ao(method="mulliken")
    print(fp.get_pop(w))

    frag1 = dft.UKS(mol1)
    frag1.xc = 'pbe0'
    frag1.kernel()

    frag2 = dft.UKS(mol2)
    frag2.xc = 'pbe0'
    frag2.kernel()

    frag_list = [frag1, frag2]
    fp = FragmentPopulation(frag_list, verbose=0)
    fp.build()
    w  = fp.make_weight_matrix_ao(method="mulliken")
    print(fp.get_pop(w))
    print(fp.get_pop(w, do_spin_pop=True))