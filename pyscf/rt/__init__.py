# Author: Junjie Yang <yangjunjie0320@gmail.com> Zheng Pei and Yihan Shao
'''
Real-Time TD-DFT calculation
======================================
Unless specified otherwise, all inputs and outputs are in atomic units(au).
Some useful conversions are:

Quantity	      Conversion
Time	        1 au = 0.02419 fs
Length	        1 au = 0.5292 A
Energy	        1 au = 27.2114 eV
Electric field	1 au = 514.2 V/nm
Dipole moment	1 au = 2.542 D
'''
import pyscf

from pyscf import rt
from pyscf import scf

from pyscf.rt import util
from pyscf.rt import rhf
from pyscf.rt import uhf
from pyscf.rt import rks
from pyscf.rt import uks

def TDHF(mf, field=None):
    if getattr(mf, 'xc', None):
        raise RuntimeError('TDHF does not support DFT object %s' % mf)
    if isinstance(mf, scf.uhf.UHF):
        mf = scf.addons.convert_to_uhf(mf)
        return uhf.TDHF(mf)
    else:
        mf = scf.addons.convert_to_rhf(mf)
        return rhf.TDHF(mf, field=field)

def TDDFT(mf, field=None):
    if isinstance(mf, scf.uhf.UHF):
        mf = scf.addons.convert_to_uhf(mf)
        if getattr(mf, 'xc', None):
            return uks.TDDFT(mf, field=field)
        else:
            return uhf.TDHF(mf, field=field)
    else:
        mf = scf.addons.convert_to_rhf(mf)
        if getattr(mf, 'xc', None):
            return rks.TDDFT(mf, field=field)
        else:
            return rhf.TDHF(mf, field=field)

TDSCF = TDDFT