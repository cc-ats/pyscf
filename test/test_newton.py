#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto
from pyscf import scf
from pyscf import soscf

mol = gto.Mole()
mol.build(
    verbose = 0,
    atom = '''8  0  0.     0
              1  0  -0.757 0.587
              1  0  0.757  0.587''',
    basis = 'ccpvdz',
)

mf = scf.RHF(mol)
mf.conv_tol = 1e-1
mf.kernel()
mo_init = mf.mo_coeff
mocc_init = mf.mo_occ

mf = scf.RHF(mol).newton()
energy = mf.kernel(mo_init, mocc_init)
print('E = %.12f, ref = -76.026765672992' % energy)
