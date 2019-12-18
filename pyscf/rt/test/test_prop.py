import numpy as np
import scipy
from functools import reduce

import pyscf
from  pyscf import dft
from  pyscf import gto
from pyscf  import rt

from pyscf.rt import print_matrix

from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

h2o =   gto.Mole( atom='''
  H   -0.0000000    0.4981795    0.7677845
  O   -0.0000000   -0.0157599    0.0000000
  H    0.0000000    0.4981795   -0.7677845
  '''
  , basis='6-31g(d)', symmetry=False).build()

h2o_rks    = dft.RKS(h2o)
h2o_rks.xc = "pbe0"

h2o_rks.max_cycle = 100
h2o_rks.conv_tol  = 1e-10

h2o_rks.verbose = 5
h2o_rks.kernel()

rttd = rt.TDSCF(h2o_rks)
rttd.chkfile = 'h2o_rt.chk'
rttd.dt = 0.2
rttd.maxstep = 10
rttd.verbose = 5
rttd.chkfile = './h2o_pbe_rttd.chk'
# rttd.efield_vec = lambda t: [0.1*np.exp(-t**2), 0.0, 0.0]
for prop_method in ['amut1', 'amut2', 'amut3', 'aeut', 'euler', 'mmut']:
    print("prop method is %s"%prop_method)
    rttd.set_prop_func(key=prop_method)
    rttd.kernel(dm_ao_init=h2o_rks.make_rdm1())

    print("ndipole = \n", rttd.ndipole)
    print("npop    = \n", rttd.npop)
    print("netot    = \n", rttd.netot - rttd.netot[0])
