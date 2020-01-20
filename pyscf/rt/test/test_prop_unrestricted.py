# Author: Junjie Yang <yangjunjie0320@gmail.com> Zheng Pei and Yihan Shao

import pyscf
from   pyscf  import dft, scf
from   pyscf  import gto
from   pyscf  import rt

h2o =   gto.Mole( atom='''
  O     0.00000000    -0.00001441    -0.34824012
  H    -0.00000000     0.76001092    -0.93285191
  H     0.00000000    -0.75999650    -0.93290797
  '''
  , basis='6-31g', symmetry=False).build()

h2o_uks    = dft.UKS(h2o)
h2o_uks.xc = "pbe0"

h2o_uks.max_cycle = 100
h2o_uks.conv_tol  = 1e-12
h2o_uks.verbose = 3
h2o_uks.kernel()
dm = h2o_uks.make_rdm1()

rttd = rt.TDSCF(h2o_uks)
rttd.dt = 0.2
rttd.maxstep = 200
rttd.verbose = 4
for prop_scheme in ["euler", "mmut", "amut1", "amut2", "amut3", "ep_pc", "lflp_pc"]:
    rttd.set_prop_func(key=prop_scheme)
    rttd.kernel(dm_ao_init=dm)

