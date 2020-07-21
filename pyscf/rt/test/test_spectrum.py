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

water =   gto.Mole( atom='''
  O     0.00000000    -0.00001441    -0.34824012
  H    -0.00000000     0.76001092    -0.93285191
  H     0.00000000    -0.75999650    -0.93290797
  '''
  , basis='6-311+g(d)', symmetry=False).build()

water_rks    = scf.RKS(water)
water_rks.verbose  = 0
water_rks.xc       = "pbe0"
water_rks.conv_tol = 1e-12
water_rks.kernel()
dm_init = water_rks.make_rdm1()

rttd = rt.TDSCF(water_rks)
rttd.total_step     = 8000
rttd.step_size      = 0.2
rttd.verbose        = 4
rttd.dm_ao_init     = dm_init
rttd.prop_method    = "mmut"
rttd.save_in_disk   = False
rttd.chk_file       = None
rttd.save_in_memory = True

field_strength = 1e-4
dipole_list = []
for i in range(3):
    temp_vec = [0.0, 0.0, 0.0]
    temp_vec[i] = field_strength
    temp_delta_vec   = lambda t: gaussian_field_vec(t, 0.0, 0.02, 0.0, temp_vec)
    temp_delta_field = ClassicalElectricField(water, field_func=temp_delta_vec, stop_time=0.5)

    rttd.electric_field = temp_delta_field
    rttd.kernel()
    
    time     = read_keyword_value("t",      result_obj=rttd.result_obj)
    temp_dip = read_keyword_value("dipole", result_obj=rttd.result_obj)
    dipole_list.append(temp_dip[:,i] - temp_dip[0,i])


mw, sigma = build_absorption_spectrum(0.2, time, numpy.array(dipole_list).T, damp_expo=50.0)

fig, ax = plt.subplots(figsize=(10,6))
lrtd = tddft.TDDFT(water_rks)
lrtd.verbose = 5
lrtd.nstates = 30
lrtd.kernel()
lrtd.analyze()
ax.stem(27.2116*lrtd.e, lrtd.oscillator_strength(), linefmt='grey', markerfmt=None, basefmt=" ", use_line_collection=True, label="LR-TDDFT")
ax.plot(27.2116*mw, 2*sigma, label="RT-TDDFT, strength=%4.2e au"%field_strength)

ax.legend(prop={'size': 10})
ax.set_title('Water Gas-Phase 6-311+G(d)/TD-PBE0 Absorption', fontsize=20)
ax.set_xlabel('Energy (eV)', fontsize=20)
ax.set_ylabel('Absorption', fontsize=20)
ax.set_xlim(0,40)
ax.set_ylim(0,1)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
ax.grid(True)
fig.tight_layout()
fig.savefig("./test_spectrum_abos.pdf")
