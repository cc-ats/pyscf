import numpy as np
import scipy.linalg as la
import copy
from functools import reduce
from pyscf import gto, scf, lo, dft, lib

class PopulationScheme(object):
    def __init__(self, mf, frgm_list):
        assert isinstance(frgm_list,      list)
        assert isinstance(frgm_list[0],   list)
        assert isinstance(frgm_list[0,0], int)

        self._scf               = mf
        self._mol               = mf.mol
        self._frgm_list         = frgm_list

        natm          = self._mol.natm
        natm_tmp      = 0
        atom_list_tmp = []
        is_uniq_atm    = False
        is_frgm_in_mol = True

        for atm_list in frgm_list:
            natm_tmp += len(frgm_list)
            for atm_idx in atm_list:
                is_uniq_atm    = atm_idx in atom_list_tmp
                is_frgm_in_mol = 0 <= atm_idx < natm
                atom_list_tmp.append(atm_idx)
        
        is_frgm_in_mol = natm_tmp == natm
        assert is_frgm_in_mol
        assert is_uniq_atm

        self.frgm_num   = len(frgm_list)
        self.atm_num    = natm
        self.bas_num    = self._mol.nbas

        if not self._scf.converged:
            mf.kernel()

    def get_frgm_list(self):
        return self._frgm_list

    def get_atm_list(self, ifrgm):
        frgm_num = self.frgm_num
        assert 0 <= ifrgm < self
        return 


    def get_weight_matrices(self):
        pass

class MullikenPopulation(PopulationScheme):
    def get_weight_matrices():
        for atm_list in self.frgm_list:

        