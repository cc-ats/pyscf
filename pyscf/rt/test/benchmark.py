# Author: Junjie Yang <yangjunjie0320@gmail.com> Zheng Pei and Yihan Shao

import pyscf
import numpy
import os
from pyscf.tools.mo_mapping import mo_comps

from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

from   pyscf  import dft, scf
from   pyscf  import gto
from   pyscf  import rt

from pyscf.rt.rhf        import kernel
from pyscf.rt.propagator import EulerPropogator
from pyscf.rt.result     import RealTimeStep, RealTimeResult

def apply_field(mol, field=[0,0,0], dm0=None):
    mf    = scf.RKS(mol)
    mf.xc = "lda"
    mf.max_cycle = 100
    h =(mol.intor('cint1e_kin_sph') + mol.intor('cint1e_nuc_sph')
        + numpy.einsum('x,xij->ij', field, mol.intor('cint1e_r_sph', comp=3)))
    mf.get_hcore = lambda *args: h
    mf.conv_tol  = 1e-10
    mf.verbose = 0
    mf.kernel(dm0)
    return mf.make_rdm1()

with open('/proc/cpuinfo') as f:
    for line in f:
        if 'model name' in line:
            print(line[:-1])
            break
with open('/proc/meminfo') as f:
    print(f.readline()[:-1])
print('OMP_NUM_THREADS=%s\n', os.environ.get('OMP_NUM_THREADS', None))


for bas in ('sto-3g', '3-21g', 'cc-pVDZ', 'cc-pVTZ', 'ANO-Roos-TZ'):
    mol = pyscf.M(atom = '''
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
''',
    symmetry=False, basis = bas)

    mf    = scf.RKS(mol)
    mf.max_cycle = 100
    mf.xc = "lda"
    mf.conv_tol  = 1e-10
    mf.verbose = 0
    mf.kernel()
    dm = mf.make_rdm1()
    dm_ = apply_field(mol, field=[0, 0, 1e-4], dm0=dm)
    print("\n    The number of basis functions: %d"%(dm.shape[0]))

    rttd = rt.TDDFT(mf)
    rttd.verbose        = 0
    rttd.total_step     = None
    rttd.step_size      = None

    prop_euler         = EulerPropogator(rttd, verbose=0)
    prop_euler.verbose = 5
    step_obj_1       = RealTimeStep(rttd,    verbose=0)
    step_obj_1.calculate_dipole = False
    step_obj_1.calculate_pop    = False
    step_obj_1.calculate_energy = False
    result_obj_1     = RealTimeResult(rttd,  verbose=3)
    result_obj_1._save_in_memory = True
    result_obj_1._save_in_disk   = False
    kernel(rttd, step_size = 0.2, total_step = 1, save_frequency = 1, dm_ao_init=dm_,
                result_obj=result_obj_1, prop_obj=prop_euler, step_obj = step_obj_1)
