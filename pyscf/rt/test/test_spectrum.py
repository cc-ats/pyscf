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

h2o =   gto.Mole( atom='''
  O     0.00000000    -0.00001441    -0.34824012
  H    -0.00000000     0.76001092    -0.93285191
  H     0.00000000    -0.75999650    -0.93290797
  '''
  , basis='6-311+g(d)', symmetry=False).build()

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
rttd.save_in_disk   = False
rttd.chk_file       = None
rttd.save_in_memory = True

field_strength = 1e-4
dipole_list = []
for i in range(3):
    temp_vec = [0.0, 0.0, 0.0]
    temp_vec[i] = field_strength
    temp_delta_vec   = lambda t: gaussian_field_vec(t, 0.0, 0.02, 0.0, temp_vec)
    temp_delta_field = ClassicalElectricField(h2o, field_func=temp_delta_vec, stop_time=0.5)

    rttd.electric_field = temp_delta_field
    rttd.kernel()
    
    time     = read_keyword_value("t",      result_obj=rttd.result_obj)
    temp_dip = read_keyword_value("dipole", result_obj=rttd.result_obj)
    dipole_list.append(temp_dip[:,i] - temp_dip[0,i])


mw, sigma = build_absorption_spectrum(0.2, time, numpy.array(dipole_list).T, damp_expo=50.0)

lrtd = tddft.TDDFT(h2o_rks)
lrtd.verbose = 4
lrtd.nstates = 30
lrtd.max_space = 100
lrtd.max_space = 200
lrtd.kernel()
lrtd.analyze()

fig, ax = plt.subplots(figsize=(10,6))
ax.stem(27.2116*lrtd.e, lrtd.oscillator_strength(), linefmt='grey', markerfmt=None, basefmt=" ", use_line_collection=True, label="LR-TDDFT")
ax.plot(27.2116*mw, 2*sigma, label="RT-TDDFT, strength=%4.2e au"%field_strength)

ax.legend(prop={'size': 10})
ax.set_title(r'H$_2$O Gas-Phase 6-311+G(d)/TD-PBE0 Absorption', fontsize=16)
ax.set_xlabel('Energy (eV)', fontsize=16)
ax.set_ylabel('Absorption', fontsize=16)
ax.set_xlim(0,50)
ax.set_ylim(0,1)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
fig.tight_layout()
fig.savefig("./test_spectrum_abos.pdf")
