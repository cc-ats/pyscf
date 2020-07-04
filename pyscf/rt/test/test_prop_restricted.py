# Author: Junjie Yang <yangjunjie0320@gmail.com> Zheng Pei and Yihan Shao

import pyscf
import numpy
from   pyscf  import dft, scf
from   pyscf  import gto
from   pyscf  import rt

from pyscf.rt.util       import print_cx_matrix
from pyscf.rt.propagator import EulerPropogator, MMUTPropogator
from pyscf.rt.propagator import EPPCPropogator, LFLPPCPropogator
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
rttd.step_size      = 0.2

prop_euler       = EulerPropogator(rttd, verbose=3)
result_obj_1     = RealTimeResult(rttd, verbose=3)
rttd.kernel(dm_ao_init=dm_, result_obj=result_obj_1, prop_obj=prop_euler)

prop_mmut        = MMUTPropogator(rttd, verbose=3)
result_obj_2     = RealTimeResult(rttd, verbose=3)
rttd.kernel(dm_ao_init=dm_, result_obj=result_obj_2, prop_obj=prop_mmut)

prop_eppc        = EPPCPropogator(rttd, verbose=3, tol=1e-12, max_iter=10)
result_obj_3     = RealTimeResult(rttd, verbose=3)
rttd.kernel(dm_ao_init=dm_, result_obj=result_obj_3, prop_obj=prop_eppc)

prop_lflp        = LFLPPCPropogator(rttd, verbose=3, tol=1e-12, max_iter=10)
result_obj_4     = RealTimeResult(rttd, verbose=3)
rttd.kernel(dm_ao_init=dm_, result_obj=result_obj_4, prop_obj=prop_lflp)

for i in range(10):
    print("")
    print("#####################################")
    print("t = %f"%result_obj_1._time_list[i])
    print("\ndelta E_\{euler\} = %e"%(result_obj_1._energy_elec_list[i]-result_obj_1._energy_elec_list[0]))
    print_cx_matrix("dm_orth = ", result_obj_1._dm_orth_list[i])
    print("\ndelta E_\{mmut\}  = %e"%(result_obj_2._energy_elec_list[i]-result_obj_2._energy_elec_list[0]))
    print_cx_matrix("dm_orth = ", result_obj_2._dm_orth_list[i])

    print("\ndelta E_\{eppc\} = %e"%(result_obj_3._energy_elec_list[i]-result_obj_3._energy_elec_list[0]))
    print_cx_matrix("dm_orth = ", result_obj_3._dm_orth_list[i])
    print("\ndelta E_\{lflp\}  = %e"%(result_obj_4._energy_elec_list[i]-result_obj_4._energy_elec_list[0]))
    print_cx_matrix("dm_orth = ", result_obj_4._dm_orth_list[i])
