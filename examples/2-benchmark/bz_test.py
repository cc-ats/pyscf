#!/usr/bin/env python
import os
import time
import re
import numpy
from pyscf import lib
from pyscf import gto, scf, dft, mcscf, mp, cc, lo

from functools import reduce

mol = gto.Mole()
mol.verbose = 5
mol.max_memory = 16000
log = lib.logger.Logger(mol.stdout, 5)
with open('/proc/cpuinfo') as f:
    for line in f:
        if 'model name' in line:
            log.note(line[:-1])
            break
with open('/proc/meminfo') as f:
    log.note(f.readline()[:-1])
log.note('OMP_NUM_THREADS=%s\n', os.environ.get('OMP_NUM_THREADS', None))

for bas in ["cc-pVDZ"]:
# for bas in ('ANO-Roos-TZ',):
    mol.atom = '''
c   1.217739890298750 -0.703062453466927  0.000000000000000
h   2.172991468538160 -1.254577209307266  0.000000000000000
c   1.217739890298750  0.703062453466927  0.000000000000000
h   2.172991468538160  1.254577209307266  0.000000000000000
c   0.000000000000000  1.406124906933854  0.000000000000000
h   0.000000000000000  2.509154418614532  0.000000000000000
c  -1.217739890298750  0.703062453466927  0.000000000000000
h  -2.172991468538160  1.254577209307266  0.000000000000000
c  -1.217739890298750 -0.703062453466927  0.000000000000000
h  -2.172991468538160 -1.254577209307266  0.000000000000000
c   0.000000000000000 -1.406124906933854  0.000000000000000
h   0.000000000000000 -2.509154418614532  0.000000000000000
'''
    mol.basis = bas
    mol.output = 'bz-%s.out' %bas
    mol.build()
    cpu0 = time.clock(), time.time()

    mf = scf.RHF(mol)
    mf.kernel()
    cpu0 = log.timer('C6H6 %s RHF'%bas, *cpu0)

    mymp2 = mp.MP2(mf)
    mymp2.kernel()
    cpu0 = log.timer('C6H6 %s MP2'%bas, *cpu0)

    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    mf.kernel()
    cpu0 = log.timer('C6H6 %s B3LYP'%bas, *cpu0)

