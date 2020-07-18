# Author: Junjie Yang <yangjunjie0320@gmail.com> Zheng Pei and Yihan Shao

import pyscf
import numpy

from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

from   pyscf  import dft, scf
from   pyscf  import gto
from   pyscf  import rt

from pyscf.rt.rhf        import kernel
from pyscf.rt.util       import print_cx_matrix
from pyscf.rt.propagator import EulerPropogator, MMUTPropogator
from pyscf.rt.propagator import EPPCPropogator, LFLPPCPropogator
from pyscf.rt.result     import RealTimeStep, RealTimeResult
from pyscf.rt.result     import read_index_list, read_step_dict, read_keyword_value

from pyscf.rt.uhf        import orth_canonical_mo, ao2orth_contravariant, ao2orth_covariant

def apply_field(mol, field=[0,0,0], dm0=None):
    mf    = scf.UKS(mol)
    mf.xc = "pbe0"
    mf.max_cycle = 100
    h =(mol.intor('cint1e_kin_sph') + mol.intor('cint1e_nuc_sph')
        + numpy.einsum('x,xij->ij', field, mol.intor('cint1e_r_sph', comp=3)))
    mf.get_hcore = lambda *args: h
    mf.conv_tol  = 1e-10
    mf.verbose = 4
    mf.kernel(dm0)
    return mf.make_rdm1()

ho =   gto.Mole( atom='''
  O    0.0000000    0.0000000    0.4951271
  H    0.0000000    0.0000000   -0.4951271
'''
, spin=1, basis='6-31g', symmetry=False).build() # OH
ho.set_common_orig([0, 0, 0])  # The gauge origin for dipole integral

ho_uks    = scf.UKS(ho)
ho_uks.max_cycle = 100
ho_uks.xc = "pbe0"
ho_uks.conv_tol  = 1e-10
ho_uks.verbose = 4
ho_uks.kernel()
dm = ho_uks.make_rdm1()
ref_dipole = ho_uks.dip_moment(unit="au")
dm_ = apply_field(ho, field=[1e-4, 0, 0], dm0=dm)

rttd = rt.TDDFT(ho_uks)
rttd.verbose        = 4
rttd.total_step     = None
rttd.step_size      = None

prop_euler       = EulerPropogator(rttd, verbose=3)
step_obj_1       = RealTimeStep(rttd,    verbose=3)
step_obj_1.calculate_dipole = True
step_obj_1.calculate_pop    = True
step_obj_1.calculate_energy = True
result_obj_1     = RealTimeResult(rttd,  verbose=3)
result_obj_1._save_in_memory = True
result_obj_1._save_in_disk   = False
kernel(rttd, step_size = 0.2, total_step = 500, save_frequency = 1, dm_ao_init=dm_,
             result_obj=result_obj_1, prop_obj=prop_euler, step_obj = step_obj_1)

t_euler         = read_keyword_value("t",           result_obj=result_obj_1)
energy_euler    = read_keyword_value("energy_elec", result_obj=result_obj_1)
dm_orth_euler   = read_keyword_value("dm_orth",     result_obj=result_obj_1)
fock_orth_euler = read_keyword_value("fock_orth",   result_obj=result_obj_1)
dipole_euler    = read_keyword_value("dipole",      result_obj=result_obj_1)

prop_mmut        = MMUTPropogator(rttd,  verbose=3)
step_obj_2       = RealTimeStep(rttd,    verbose=3)
step_obj_2.calculate_dipole = True
step_obj_2.calculate_pop    = True
step_obj_2.calculate_energy = True
result_obj_2     = RealTimeResult(rttd,  verbose=3)
result_obj_2._save_in_memory = True
result_obj_2._save_in_disk   = False
kernel(rttd, step_size = 0.2, total_step = 500, save_frequency = 1, dm_ao_init=dm_,
             result_obj=result_obj_2, prop_obj=prop_mmut, step_obj = step_obj_2)

t_mmut         = read_keyword_value("t",           result_obj=result_obj_2)
energy_mmut    = read_keyword_value("energy_elec", result_obj=result_obj_2)
dm_orth_mmut   = read_keyword_value("dm_orth",     result_obj=result_obj_2)
fock_orth_mmut = read_keyword_value("fock_orth",   result_obj=result_obj_2)
dipole_mmut    = read_keyword_value("dipole",      result_obj=result_obj_2)

prop_eppc        = EPPCPropogator(rttd,  verbose=3)
step_obj_3       = RealTimeStep(rttd,    verbose=3)
prop_eppc.tol    = 1e-12
step_obj_3.calculate_dipole = True
step_obj_3.calculate_pop    = True
step_obj_3.calculate_energy = True
result_obj_3     = RealTimeResult(rttd,  verbose=3)
result_obj_3._save_in_memory = True
result_obj_3._save_in_disk   = False
kernel(rttd, step_size = 0.2, total_step = 500, save_frequency = 1, dm_ao_init=dm_,
             result_obj=result_obj_3, prop_obj=prop_eppc, step_obj = step_obj_3)

t_eppc         = read_keyword_value("t",            result_obj=result_obj_3)
energy_eppc    = read_keyword_value("energy_elec",  result_obj=result_obj_3)
dm_orth_eppc   = read_keyword_value("dm_orth",      result_obj=result_obj_3)
fock_orth_eppc = read_keyword_value("fock_orth",    result_obj=result_obj_3)
dipole_eppc    = read_keyword_value("dipole",       result_obj=result_obj_3)

prop_lflp        = LFLPPCPropogator(rttd, verbose=3)
step_obj_4       = RealTimeStep(rttd,     verbose=3)
prop_lflp.tol    = 1e-12
step_obj_4.calculate_dipole = True
step_obj_4.calculate_pop    = True
step_obj_4.calculate_energy = True
result_obj_4     = RealTimeResult(rttd,  verbose=3)
result_obj_4._save_in_memory = True
result_obj_4._save_in_disk   = False
kernel(rttd, step_size = 0.2, total_step = 500, save_frequency = 1, dm_ao_init=dm_,
             result_obj=result_obj_4, prop_obj=prop_lflp, step_obj = step_obj_4)

t_lflp         = read_keyword_value("t",            result_obj=result_obj_4)
energy_lflp    = read_keyword_value("energy_elec",  result_obj=result_obj_4)
dm_orth_lflp   = read_keyword_value("dm_orth",      result_obj=result_obj_4)
fock_orth_lflp = read_keyword_value("fock_orth",    result_obj=result_obj_4)
dipole_lflp    = read_keyword_value("dipole",       result_obj=result_obj_4)   

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(t_euler, energy_euler - energy_euler[0], label="Euler")
ax1.plot(t_mmut,  energy_mmut  -  energy_mmut[0], label="MMUT")
ax1.plot(t_eppc,  energy_eppc  -  energy_eppc[0], label="EPPC")
ax1.plot(t_lflp,  energy_lflp  -  energy_lflp[0], label="LFLP")
ax1.legend()
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
ax1.set_ylim(
   -numpy.max(numpy.abs(energy_mmut  -  energy_mmut[0]))*10.0,
    numpy.max(numpy.abs(energy_mmut  -  energy_mmut[0]))*10.0
)

ax2.plot(t_euler,dipole_euler[:,0] -   ref_dipole[0], label="Euler")
ax2.plot(t_mmut,  dipole_mmut[:,0]  -  ref_dipole[0], label="MMUT")
ax2.plot(t_eppc,  dipole_eppc[:,0]  -  ref_dipole[0], label="EPPC")
ax2.plot(t_lflp,  dipole_lflp[:,0]  -  ref_dipole[0], label="LFLP")
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))

ax1.set_title(r"HO Gas-Phase 6-31G/TD-PBE0")
ax2.set_xlabel('Time (au)')
ax1.set_ylabel('Energy Error (au)')
ax2.set_ylabel('Z-Dipole (au)')
plt.tight_layout()
fig.savefig("./test_prop_unrestricted.pdf")