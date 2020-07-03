# Author: Junjie Yang <yangjunjie0320@gmail.com> Zheng Pei and Yihan Shao

import pyscf
import numpy
from   pyscf  import dft, scf
from   pyscf  import gto
from   pyscf  import rt

from pyscf.rt.util       import print_cx_matrix
from pyscf.rt.propagator import EulerPropogator
from pyscf.rt.result     import RealTimeStep, RealTimeResult

h2o =   gto.Mole( atom='''
O     0.00000000    -0.00001441    -0.34824012
H    -0.00000000     0.76001092    -0.93285191
H     0.00000000    -0.75999650    -0.93290797
'''
, basis='sto-3g', symmetry=False).build() # water
h2o.set_common_orig([0, 0, 0])  # The gauge origin for dipole integral

h2o_rhf    = scf.RHF(h2o)
# h2o_rhf.xc = "pbe0"
h2o_rhf.max_cycle = 10
h2o_rhf.conv_tol  = 1e-12
h2o_rhf.verbose = 4
h2o_rhf.kernel()
dm = h2o_rhf.make_rdm1()

h2o_rhf_    = scf.RHF(h2o)
# h2o_rhf.xc = "pbe0"
h2o_rhf_.max_cycle = 10
h =(h2o.intor('cint1e_kin_sph') + h2o.intor('cint1e_nuc_sph')
    + numpy.einsum('x,xij->ij', [1e-2,0.0,0.0], h2o.intor('cint1e_r_sph', comp=3)))
h2o_rhf_.get_hcore = lambda *args: h
h2o_rhf_.conv_tol  = 1e-12
h2o_rhf_.verbose = 4
h2o_rhf_.kernel(dm)
dm_ = h2o_rhf_.make_rdm1(dm0=dm)

rttd = rt.TDHF(h2o_rhf)
rttd.verbose        = 3
rttd.total_step     = 10
rttd.step_size      = 0.02
# rttd.prop_obj       = EulerPropogator(rttd, verbose=3)
# rttd.step_obj       = RealTimeStep(rttd, verbose=3)
# rttd.result_obj     = RealTimeResult(rttd, verbose=3)
# rttd._initialize()
rttd.kernel(dm_ao_init=dm_)

for i in range(10):
    print("")
    print("#####################################")
    print("t = %f"%rttd.result_obj._time_list[i])
    print_cx_matrix("dm_orth = ", rttd.result_obj._dm_orth_list[i])
    print_cx_matrix("fock_orth = ", rttd.result_obj._fock_orth_list[i])