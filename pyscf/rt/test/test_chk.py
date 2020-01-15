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

h2o_rks    = dft.UKS(h2o)
h2o_rks.xc = "pbe0"

h2o_rks.max_cycle = 100
h2o_rks.conv_tol  = 1e-12
h2o_rks.verbose = 3
h2o_rks.kernel()
dm = h2o_rks.make_rdm1()

rttd = rt.TDSCF(h2o_rks)
rttd.dt = 0.2
rttd.maxstep = 100
rttd.verbose = 4
rttd.chkfile = './test_chk.chk'
rttd.save_step = 3
rttd.set_prop_func(key="mmut")
rttd.kernel(dm_ao_init=dm)
print("rttd.ntime shape is", rttd.ntime.shape)
print("rttd shape is\n", rttd.ntime)
print("rttd.ndm_prim shape is", rttd.ndm_prim.shape)

rttd.__dict__.update(
  scf.chkfile.load('./test_chk.chk', 'rt')
  )

print("rttd.ntime shape is", rttd.ntime.shape)
print("rttd shape is\n", rttd.ntime)
print("rttd.ndm_prim shape is", rttd.ndm_prim.shape)