# Author: Junjie Yang <yangjunjie0320@gmail.com> Zheng Pei and Yihan Shao
import pyscf

from pyscf import rt
from pyscf import scf

from pyscf.rt import util
from pyscf.rt import chkfile

from pyscf.rt import rhf
from pyscf.rt import uhf
from pyscf.rt import rks
from pyscf.rt import uks

def TDHF(mf):
    if getattr(mf, 'xc', None):
        raise RuntimeError('TDHF does not support DFT object %s' % mf)
    if isinstance(mf, scf.uhf.UHF):
        mf = scf.addons.convert_to_uhf(mf)
        return uhf.TDHF(mf)
    else:
        mf = scf.addons.convert_to_rhf(mf)
        return rhf.TDHF(mf)

def TDDFT(mf):
    if isinstance(mf, scf.uhf.UHF):
        mf = scf.addons.convert_to_uhf(mf)
        if getattr(mf, 'xc', None):
            return uks.TDDFT(mf)
        else:
            return uhf.TDHF(mf)
    else:
        mf = scf.addons.convert_to_rhf(mf)
        if getattr(mf, 'xc', None):
            return rks.TDDFT(mf)
        else:
            return rhf.TDHF(mf)

TDSCF = TDDFT