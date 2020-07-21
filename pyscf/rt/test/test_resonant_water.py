# Author: Junjie Yang <yangjunjie0320@gmail.com> Zheng Pei and Yihan Shao

import pyscf
import numpy
from numpy import exp, cos, sin, power

from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

from   pyscf  import dft, scf, tddft
from   pyscf  import gto
from   pyscf  import rt

from pyscf.rt.util   import build_absorption_spectrum
from pyscf.rt.field  import ClassicalElectricField
from pyscf.rt.result import read_index_list, read_step_dict, read_keyword_value


h2o =   gto.Mole( atom='''
  O     0.00000000    -0.00001441    -0.34824012
  H    -0.00000000     0.76001092    -0.93285191
  H     0.00000000    -0.75999650    -0.93290797
  '''
  , basis='6-311+g(d)', symmetry=False).build()

h2o_rks          = scf.RKS(h2o)
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
rttd.save_in_disk   = False
rttd.chk_file       = None
rttd.save_in_memory = True

field_strength = 1e-4
freq           = 0.28902542
period         = 2*numpy.pi/freq
sin_vec_z      = lambda t: [1e-4*sin(freq*t), 0.0, 0.0]
sin_field_z    = ClassicalElectricField(h2o, field_func=sin_vec_z, stop_time=8.0*period)
rttd.electric_field = sin_field_z
rttd.kernel()

time = read_keyword_value("t",      result_obj=rttd.result_obj)
dip  = read_keyword_value("dipole", result_obj=rttd.result_obj)
dxx  = dip[:,0] - dip[0,0]
mw, sigma = build_absorption_spectrum(0.2, time[:7*rttd.total_step//8], dip[1*rttd.total_step//8+1:,:], damp_expo=50.0)

lrtd = tddft.TDDFT(h2o_rks)
lrtd.verbose = 4
lrtd.nstates = 30
lrtd.max_space = 100
lrtd.max_space = 200
lrtd.kernel()
lrtd.analyze()

fig, (ax1,ax2) = plt.subplots(2,1,figsize=(10,10))
ax1.stem(27.2116*lrtd.e, lrtd.oscillator_strength(), linefmt='grey', markerfmt=None, basefmt=" ", use_line_collection=True, label="LR-TDDFT, Oscillator Strength")
ax1.plot(27.2116*mw, sigma, label="RT-TDDFT, Freq=%4.4f au, Strength=%4.2e au"%(freq, field_strength))

ax1.legend(prop={'size': 10})
ax1.set_title(r'H$_2$O Gas-Phase 6-311+G(d)/TD-PBE0 Absorption', fontsize=16)
ax1.set_xlabel('Energy (eV)', fontsize=16)
ax1.set_ylabel('Absorption (Scaled)', fontsize=16)
ax1.set_xlim(0,50)
ax1.set_ylim(0,1)
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))

right_axis = ax2.twinx()
left_axis = ax2
p1, = left_axis.plot( time, dxx, label="X-Dipole",color='C0' )
p_, = left_axis.plot(-time, dxx, label="Electric Field",   color='C1' )
p2, = right_axis.plot(time, [sin_field_z.get_field_vec(t)[0] for t in time], label="Field",color='C1')
 
left_axis.set_xlabel('Time (fs)',   fontsize=16)
left_axis.set_ylabel('Dipole (au)', fontsize=16)
right_axis.set_ylabel('Field (au)', fontsize=16)
right_axis.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
left_axis.set_ylim(-0.016,0.016)
left_axis.set_xlim(0,1600)
left_axis.legend(prop={'size': 10})
fig.tight_layout()
fig.savefig("./test_resonant_water.pdf")
