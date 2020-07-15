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

def delta_efield(t):
    return 0.0001*numpy.exp(-10*t**2/0.2**2)

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

gaussian_vec_x   = lambda t: gaussian_field_vec(t, 0.0, 0.02, 0.0, [1e-4, 0.0, 0.0])
gaussian_field_x = ClassicalElectricField(h2o, field_func=gaussian_vec_x, stop_time=0.5)

gaussian_vec_y   = lambda t: gaussian_field_vec(t, 0.0, 0.02, 0.0, [0.0, 1e-4, 0.0])
gaussian_field_y = ClassicalElectricField(h2o, field_func=gaussian_vec_x, stop_time=0.5)

gaussian_vec_z   = lambda t: gaussian_field_vec(t, 0.0, 0.02, 0.0, [0.0, 0.0, 1e-4])
gaussian_field_z = ClassicalElectricField(h2o, field_func=gaussian_vec_x, stop_time=0.5)

rttd = rt.TDSCF(h2o_rks)
rttd.total_step     = 5000
rttd.step_size      = 0.2
rttd.verbose        = 4
rttd.dm_ao_init     = dm_init
rttd.prop_method    = "eppc"

rttd.save_in_disk   = True
rttd.chk_file       = "h2o_x.chk"
rttd.save_in_memory = False
rttd.electric_field = gaussian_field_x
rttd.kernel()
time = read_keyword_value("t", chk_file="h2o_x.chk")
dip1 = read_keyword_value("dipole", chk_file="h2o_x.chk")
dxx  = dip1[:,0] - dip1[0,0]

rttd.chk_file       = "h2o_y.chk"
rttd.electric_field = gaussian_field_y
rttd.kernel()
dip2 = read_keyword_value("dipole", chk_file="h2o_y.chk")
dyy  = dip2[:,0] - dip2[0,0]

rttd.chk_file       = "h2o_z.chk"
rttd.electric_field = gaussian_field_z
rttd.kernel()
dip3 = read_keyword_value("dipole", chk_file="h2o_z.chk")
dzz  = dip3[:,0] - dip3[0,0]

dipole_fig, (dipole_axes1, dipole_axes2, dipole_axes3) = plt.subplots(3, 1, sharex=True, figsize=(20,20))
dipole_axes1.set_ylabel('XX-Dipole (au)', fontsize=20)
dipole_axes1.set_title("Water Gas-Phase 6-31G/TD-PBE0 Dipole")
dipole_axes1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%5.2e'))
dipole_axes1.grid(True)

dipole_axes2.set_ylabel('YY-Dipole (au)', fontsize=20)
dipole_axes2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%5.2e'))
dipole_axes2.grid(True)

dipole_axes3.set_xlabel('time (fs)', fontsize=20)
dipole_axes3.set_ylabel('ZZ-Dipole (au)', fontsize=20)
dipole_axes3.yaxis.set_major_formatter(ticker.FormatStrFormatter('%5.2e'))
dipole_axes3.grid(True)

dipole_axes1.plot(0.02419*rttd.ntime, dxx)
dipole_axes2.plot(0.02419*rttd.ntime, dyy)
dipole_axes3.plot(0.02419*rttd.ntime, dzz)
dipole_fig.savefig("./h2o_dip.pdf")

lrtd = tddft.TDDFT(h2o_rks)
lrtd.nstates = 30
lrtd.kernel()
mw, sigma = build_absorption_spectrum(0.2, time, numpy.array([dxx,dyy,dzz]).T, damp_expo=50.0)

fig, ax = plt.subplots(figsize=(20,10))
ax.plot(27.2116*mw, sigma, label="rt-tddft")
ax.stem(27.2116*lrtd.e, lrtd.oscillator_strength(), linefmt='grey', markerfmt=None, basefmt=" ", use_line_collection=True, label="lr-tddft")
ax.legend()
ax.set_title("Water Gas-Phase 6-31G/TD-PBE0 Absorption", fontsize=20)
ax.set_xlabel('Energy (eV)', fontsize=20)
ax.set_ylabel('Absorption', fontsize=20)
ax.set_xlim(0,25)
ax.set_ylim(0,1)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
ax.grid(True)
fig.savefig("./h2o_abos.pdf")
