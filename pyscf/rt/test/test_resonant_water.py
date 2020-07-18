# Author: Junjie Yang <yangjunjie0320@gmail.com> Zheng Pei and Yihan Shao

import pyscf
import numpy

from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

from   pyscf  import dft, scf, tddft
from   pyscf  import gto
from   pyscf  import rt

from pyscf.rt.util   import build_absorption_spectrum
from pyscf.rt.field  import ClassicalElectricField, constant_field_vec, gaussian_field_vec
from pyscf.rt.result import read_index_list, read_step_dict, read_keyword_value

def z_oscillator_strength(tdobj, e=None, xy=None, gauge='length', order=0):
    if e is None: e = tdobj.e

    if gauge == 'length':
        trans_dip = tdobj.transition_dipole(xy=xy)
        f = numpy.einsum('s,s,s->s', e, trans_dip[:,2], trans_dip[:,2])
        return f/numpy.linalg.norm(f)


h2o =   gto.Mole( atom='''
  O     0.00000000    -0.00001441    -0.34824012
  H    -0.00000000     0.76001092    -0.93285191
  H     0.00000000    -0.75999650    -0.93290797
  '''
  , basis='6-31g', symmetry=False).build()

h2o_rks    = scf.RKS(h2o)
h2o_rks.verbose  = 0
h2o_rks.xc       = "pbe0"
h2o_rks.conv_tol = 1e-12
h2o_rks.kernel()
dm_init = h2o_rks.make_rdm1()

rttd = rt.TDSCF(h2o_rks)
rttd.total_step     = 8000
rttd.step_size      = 0.2
rttd.verbose        = 4
rttd.dm_ao_init     = dm_init
rttd.prop_method    = "mmut"

lrtd = tddft.TDDFT(h2o_rks)
lrtd.nstates = 30
lrtd.kernel()

fig, (ax1,ax2) = plt.subplots(2,1,figsize=(10,10))
ax1.stem(27.2116*lrtd.e, z_oscillator_strength(lrtd), linefmt='grey', markerfmt=None, basefmt=" ", use_line_collection=True, label="LR-TDDFT, z oscillator strength")

for field_strength in [5e-3, 1e-3, 2e-4]:
    cos_vec_z   = lambda t: numpy.cos(0.3768*t)*numpy.asarray([0.0, 0.0, field_strength])
    cos_field_z = ClassicalElectricField(h2o, field_func=cos_vec_z, stop_time=100.0)

    rttd.save_in_disk   = True
    rttd.chk_file       = "h2o_z_%.2e.chk"%field_strength
    rttd.save_in_memory = False
    rttd.electric_field = cos_field_z
    # rttd.kernel()
    time = read_keyword_value("t",      chk_file="h2o_z_%.2e.chk"%field_strength)
    dip  = read_keyword_value("dipole", chk_file="h2o_z_%.2e.chk"%field_strength)
    dzz  = dip[:,2] - dip[0,2]

    mw, sigma = build_absorption_spectrum(0.2, time[:7*rttd.total_step//8], dip[1*rttd.total_step//8+1:,:], damp_expo=50.0)
    ax1.plot(27.2116*mw, sigma, label="RT-TDDFT, field strength=%.2e au"%field_strength)

ax1.legend()
ax1.set_title("Water Gas-Phase 6-31G/TD-PBE0", fontsize=24)
ax1.set_xlabel('Energy (eV)', fontsize=16)
ax1.set_ylabel('Absorption', fontsize=16)
ax1.set_xlim(0,30)
ax1.set_ylim(0,1)
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
ax1.grid(True)

right_axis = ax2.twinx()
left_axis = ax2
p1, = left_axis.plot(time, dzz, label="Z-Dipole",color='C0' )
p_, = left_axis.plot(-time, dzz, label="Field",color='C1' )
p2, = right_axis.plot(time, [cos_field_z.get_field_vec(t)[2] for t in time], label="Field",color='C1')
 
left_axis.set_xlabel('Time (fs)', fontsize=16)
left_axis.set_ylabel('Dipole (au)', fontsize=16)
right_axis.set_ylabel('Field (au)', fontsize=16)
right_axis.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
left_axis.set_ylim(-0.016,0.016)
left_axis.set_xlim(0,1600)
left_axis.legend()
ax2.grid(True)
fig.tight_layout()
fig.savefig("./test_resonant_water.pdf")