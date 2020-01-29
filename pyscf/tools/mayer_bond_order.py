# Author: Junjie Yang <yangjunjie0320@gmail.com> Zheng Pei and Yihan Shao

import time
import numpy
import pyscf
from pyscf import lib
from pyscf.lib import logger

from pyscf import gto
from pyscf import scf
from pyscf import dft
from pyscf.dft import numint
from pyscf.tools import cubegen

import copy
from copy import deepcopy
from functools import reduce

from pyscf import __config__

def mayer_bond_order(mf, atm_idx1, atm_idx2, ovlp=None, dm=None):
    if ovlp is None:
        ovlp = mf.get_ovlp()
    if dm is None:
        dm = mf.make_rdm1()
    [[s1, e1], [s2, e2]] = mf.mol.aoslice_by_atom()[[atm_idx1, atm_idx2], 2:4]
    mayerb1 = numpy.einsum('mu,un->mn', dm[s1:e1,:], ovlp[:,s2:e2])
    mayerb2 = numpy.einsum('mu,un->mn', dm[s2:e2,:], ovlp[:,s1:e1])
    mayer_bo = numpy.einsum('mu,um->', mayerb1, mayerb2)
    print('mayer bond order: ', mayer_bo)
    return mayer_bo

if __name__ == "__main__":
    mol = gto.Mole()
    mol.atom = '''
    C         0.6584188440    0.0000000000    0.0000000000
    C        -0.6584188440    0.0000000000    0.0000000000
    H         1.2256624142    0.9143693597    0.0000000000
    H         1.2256624142   -0.9143693597    0.0000000000
    H        -1.2256624142    0.9143693597    0.0000000000
    H        -1.2256624142   -0.9143693597    0.0000000000'''
    mol.basis = 'cc-pvdz'
    mol.build()
    mf = scf.RKS(mol)
    mf.xc = "PBE"
    mf.kernel()
    mayer_bond_order(mf, 0, 1)
    mayer_bond_order(mf, 0, 2)
    mayer_bond_order(mf, 1, 2)