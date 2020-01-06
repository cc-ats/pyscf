import numpy as np
import scipy
from functools import reduce

import pyscf
from  pyscf import dft
from  pyscf import gto
from pyscf  import rt

def delta_efield(t):
    return 0.0001*np.exp(-10*t**2/0.2**2)

h2o =   gto.Mole( atom='''
  H   -0.0000000    0.4981795    0.7677845
  O   -0.0000000   -0.0157599    0.0000000
  H    0.0000000    0.4981795   -0.7677845
  '''
  , basis='6-31g', symmetry=False).build()

h2o_rks    = dft.RKS(h2o)
h2o_rks.xc = "pbe0"

h2o_rks.max_cycle = 100
h2o_rks.conv_tol  = 1e-10

h2o_rks.verbose = 5
h2o_rks.kernel()
dm = h2o_rks.make_rdm1()

rttd = rt.TDSCF(h2o_rks)
rttd.set_prop_func(key="amut3")
rttd.efield_vec = lambda t: [delta_efield(t), 0.0, 0.0]
rttd.chkfile = 'h2o_rt_x.chk'
rttd.dt = 0.2
rttd.maxstep = 1000
rttd.verbose = 4
rttd.kernel(dm_ao_init=dm)

rttd.efield_vec = lambda t: [0.0, delta_efield(t), 0.0]
rttd.chkfile = 'h2o_rt_y.chk'
rttd.dt = 0.2
rttd.maxstep = 1000
rttd.verbose = 4
rttd.kernel(dm_ao_init=dm)

rttd.efield_vec = lambda t: [0.0, 0.0, delta_efield(t)]
rttd.chkfile = 'h2o_rt_z.chk'
rttd.dt = 0.2
rttd.maxstep = 1000
rttd.verbose = 4
rttd.kernel(dm_ao_init=dm)