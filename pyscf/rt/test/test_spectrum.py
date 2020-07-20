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

adenine =   gto.Mole( atom='''
  N     -2.2027     -0.0935     -0.0068
  C     -0.9319      0.4989     -0.0046
  C      0.0032     -0.5640     -0.0064
  N     -0.7036     -1.7813     -0.0164
  C     -2.0115     -1.4937     -0.0151
  C      1.3811     -0.2282     -0.0017
  N      1.7207      1.0926     -0.0252
  C      0.7470      2.0612     -0.0194
  N     -0.5782      1.8297     -0.0072
  H      1.0891      3.1044     -0.0278
  N      2.4103     -1.1617      0.1252
  H     -3.0709      0.3774     -0.0035
  H     -2.8131     -2.2379     -0.0191
  H      2.1765     -2.0715     -0.1952
  H      3.3110     -0.8521     -0.1580
  '''
  , basis='6-311+g(d)', symmetry=False).build()

adenine_rks    = scf.RKS(adenine)
adenine_rks.verbose  = 0
adenine_rks.xc       = "pbe0"
adenine_rks.conv_tol = 1e-12
adenine_rks.kernel()
dm_init = adenine_rks.make_rdm1()

rttd = rt.TDSCF(adenine_rks)
rttd.total_step     = 8000
rttd.step_size      = 0.2
rttd.verbose        = 4
rttd.dm_ao_init     = dm_init
rttd.prop_method    = "mmut"

lrtd = tddft.TDDFT(adenine_rks)
lrtd.verbose = 4
lrtd.nstates = 30
lrtd.kernel()

fig, ax = plt.subplots(figsize=(10,6))
ax.stem(27.2116*lrtd.e, lrtd.oscillator_strength(), linefmt='grey', markerfmt=None, basefmt=" ", use_line_collection=True, label="LR-TDDFT")

field_strength = 1e-4
gaussian_vec_x   = lambda t: gaussian_field_vec(t, 0.0, 0.02, 0.0, [field_strength, 0.0, 0.0])
gaussian_field_x = ClassicalElectricField(adenine, field_func=gaussian_vec_x, stop_time=0.5)

gaussian_vec_y   = lambda t: gaussian_field_vec(t, 0.0, 0.02, 0.0, [0.0, field_strength, 0.0])
gaussian_field_y = ClassicalElectricField(adenine, field_func=gaussian_vec_y, stop_time=0.5)

gaussian_vec_z   = lambda t: gaussian_field_vec(t, 0.0, 0.02, 0.0, [0.0, 0.0, field_strength])
gaussian_field_z = ClassicalElectricField(adenine, field_func=gaussian_vec_z, stop_time=0.5)

rttd.save_in_disk   = False
rttd.chk_file       = None
rttd.save_in_memory = True

rttd.electric_field = gaussian_field_x
rttd.kernel()
time = read_keyword_value("t",      result_obj=rttd.result_obj)
dip1 = read_keyword_value("dipole", result_obj=rttd.result_obj)
dxx  = dip1[:,0] - dip1[0,0]

rttd.electric_field = gaussian_field_y
rttd.kernel()
dip2 = read_keyword_value("dipole", result_obj=rttd.result_obj)
dyy  = dip2[:,1] - dip2[0,1]

rttd.electric_field = gaussian_field_z
rttd.kernel()
dip3 = read_keyword_value("dipole", result_obj=rttd.result_obj)
dzz  = dip3[:,2] - dip3[0,2]

mw, sigma = build_absorption_spectrum(0.2, time, numpy.array([dxx,dyy,dzz]).T, damp_expo=50.0)
ax.plot(27.2116*mw, sigma, label="RT-TDDFT, strength=%e au"%field_strength)

ax.legend(prop={'size': 10})
ax.set_title("Adenine Gas-Phase 6-311+G(d)/TD-PBE0 Absorption", fontsize=20)
ax.set_xlabel('Energy (eV)', fontsize=20)
ax.set_ylabel('Absorption', fontsize=20)
ax.set_xlim(0,40)
ax.set_ylim(0,1)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
ax.grid(True)
fig.tight_layout()
fig.savefig("./test_spectrum_abos.pdf")

