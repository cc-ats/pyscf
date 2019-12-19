import time
import numpy
import pyscf
from pyscf import lib
from pyscf.lib import logger

from pyscf import gto
from pyscf import scf
from pyscf import dft
from pyscf.dft import numint
from pyscf.tools import cubegen, energy_density

import copy
from copy import deepcopy
from functools import reduce

from pyscf import __config__

MUTE_CHKFILE     = getattr(__config__, 'mute_chkfile', False)
NELEC_ERROR_TOL  = getattr(__config__, 'dft_rks_prune_error_tol', 0.02)
SMALL_RHO_CUTOFF = getattr(__config__, 'dft_rks_RKS_small_rho_cutoff', 1e-7)

class FragmentMethod(lib.StreamObject):
    def __init__(self, mf_list, verbose=5):
        self.verbose = verbose
        self.nfrg = len(mf_list)
        self.frag_list = mf_list

        self.mol = reduce( gto.mole.conc_mol, [mf.mol for mf in self.frag_list] )
        self.mf  = self.frag_list[0].__class__(self.mol)
        self.mf.max_cycle = self.frag_list[0].max_cycle
        if hasattr(self.frag_list[0], "xc"):
            self.mf.xc = self.frag_list[0].xc
        self.mf.conv_tol = self.frag_list[0].conv_tol
        self.mf.verbose = 4


    def build(self):
        self.mf.kernel()
        self.grid = dft.gen_grid.Grids(self.mol)
        self.grid.build()
        self.grid.coords, self.grid.weights, self.grid.non0tab = self._prune_small_rho_grids(
        dm=self.mf.make_rdm1(), coords=self.grid.coords, weights=self.grid.weights
        )

        temp_index = list(range(self.mol.natm))
        self.frag_idx = []
        count = 0
        for frag in [frag_.mol for frag_ in self.frag_list]:
            self.frag_idx.append(temp_index[count:count+frag.natm])
            count = count+frag.natm

        logger.info(self, '')
        if hasattr(self.mf, 'xc'):
            logger.info(self, 'exchange correlation functional is %s', self.mf.xc)
        
        for imf, frag_mf in enumerate(self.frag_list):
            logger.info(self, 'the fragment %d energy is %f', imf, frag_mf.e_tot)
        logger.info(self, 'the total energy      is %f', self.mf.e_tot)
        logger.info(self, 'the energy difference is %e (without BSSE)', # TODO: BSSE method
        self.mf.e_tot - numpy.sum([frag_mf.e_tot for frag_mf in self.frag_list])
        )
        logger.info(self, '')

    def _prune_small_rho_grids(self, dm=None, coords=None, weights=None, non0tab=None):
        if dm is None:
            dm = self.mf.make_rdm1()
        grid = self.grid
        if coords is None:
            coords = grid.coords
        if weights is None:
            weights = grid.weights
        if non0tab is None:
            non0tab = grid.non0tab
        ao_value = numint.eval_ao(self.mol, coords, deriv=0)
        rho = numint.eval_rho(self.mol, ao_value, dm, xctype='lda')
        n = numpy.dot(rho, weights)
        mol = self.mol
        size = weights.size
        if abs(n-mol.nelectron) < NELEC_ERROR_TOL*n:
            rho *= weights
            idx = abs(rho) > SMALL_RHO_CUTOFF / weights.size
            self.drop_idx = abs(rho) <= SMALL_RHO_CUTOFF / weights.size
            coords  = numpy.asarray(coords [idx], order='C')
            weights = numpy.asarray(weights[idx], order='C')
            non0tab = grid.make_mask(mol, coords)
        logger.info(self, 'Drop grid %d',
                        size - numpy.count_nonzero(idx))
        return coords, weights, non0tab

    def build_fbh_weight(self):
        if not hasattr(self, "fbh_weight"):
            frag_ao_value_list = [numint.eval_ao(frag.mol, self.grid.coords, deriv=0) for frag in self.frag_list]
            frag_rho_list = []
            for imf, frag_mf in enumerate(self.frag_list):
                frag_rho_list.append(
                    numint.eval_rho(frag_mf.mol, frag_ao_value_list[imf], frag_mf.make_rdm1(), xctype='lda')
                )
            frag_rho_list = numpy.array(frag_rho_list)
            self.fbh_weight = frag_rho_list/numpy.einsum("ij->j", frag_rho_list)
            return self.fbh_weight
        else:
            return self.fbh_weight

    def build_becke_weight(self):
        if not hasattr(self, "becke_weight"):
            ngrids = self.grid.coords.shape[0]
            atomic_ngrid_ = [0]
            atom_grids_tab = self.grid.gen_atomic_grids(self.mol)
            for iatm in range(self.mol.natm):
                dcoords, weight = atom_grids_tab[self.mol.atom_symbol(iatm)]
                atomic_ngrid_.append(weight.size)
            atomic_ngrid = numpy.add.accumulate(atomic_ngrid_)
            for iatm in range(self.mol.natm):
                atomic_ngrid_[iatm+1] = atomic_ngrid_[iatm+1] - numpy.count_nonzero(self.drop_idx[atomic_ngrid[iatm]:atomic_ngrid[iatm+1]])
            atomic_ngrid = numpy.add.accumulate(atomic_ngrid_)

            self.becke_weight = numpy.zeros([self.nfrg, ngrids])
            for ifrg in range(self.nfrg):
                for iatm in self.frag_idx[ifrg]:
                    self.becke_weight[ifrg,atomic_ngrid[iatm]:atomic_ngrid[iatm+1]] += 1.0
            return self.becke_weight
        else:
            return self.becke_weight

    def real_space_func_partition(self, rho, method='FBH'):
        if method.lower() == 'fbh':
            fbh_weight = self.build_fbh_weight()
            frag_partition = numpy.einsum("i,i,ji->j", self.grid.weights, rho, fbh_weight)
            return frag_partition
        elif method.lower() == 'becke':
            becke_weight = self.build_becke_weight()
            frag_partition = numpy.einsum("i,i,ji->j", self.grid.weights, rho, becke_weight)
            return frag_partition




if __name__ == "__main__":
    frag1 = gto.Mole()
    frag1.atom = '''
O        -1.5191160500    0.1203799285    0.0000000000
H        -1.9144370615   -0.7531546521    0.0000000000
H        -0.5641810316   -0.0319363903    0.0000000000'''
    frag1.basis = '6-31g(d)'
    frag1.build()
    mf1 = dft.RKS(frag1)
    mf1.xc = 'pbe0'
    mf1.kernel()

    frag2 = gto.Mole()
    frag2.atom = '''
O         1.3908332592   -0.1088624008    0.0000000000
H         1.7527099026    0.3464799298   -0.7644430086
H         1.7527099026    0.3464799298    0.7644430086'''
    frag2.basis = '6-31g(d)'
    frag2.build()
    mf2 = dft.RKS(frag2)
    mf2.xc = 'pbe0'
    mf2.kernel()

    frag_list = [frag1, frag2]
    mf_list   = [mf1,   mf2  ]
    fm = FragmentMethod(mf_list, verbose=5)
    fm.build()

    ao_value = numint.eval_ao(fm.mol, fm.grid.coords, deriv=2)
    rho      = numint.eval_rho(fm.mol, ao_value, fm.mf.make_rdm1(), xctype='mGGA')

    elec_den = rho[0]
    ener_den = energy_density.calc_rho_ene(fm.mf, fm.grid.coords, fm.mf.make_rdm1(), ao_value=ao_value)
    print(fm.mol.nelec)
    print("FBH density partition",   fm.real_space_func_partition(elec_den, method='fbh'))
    print("Becke density partition", fm.real_space_func_partition(elec_den, method='becke'))
    print("FBH energy partition",    fm.real_space_func_partition(ener_den, method='fbh'))
    print("Becke energy partition",  fm.real_space_func_partition(ener_den, method='becke'))





