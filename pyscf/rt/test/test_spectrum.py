import numpy
import scipy
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

from functools import reduce
from time      import time

import pyscf
from   pyscf  import dft, scf
from   pyscf  import gto
from   pyscf  import rt, tddft

def delta_efield(t):
    return 0.0001*numpy.exp(-10*t**2/0.2**2)

h2o =   gto.Mole( atom='''
  O     0.00000000    -0.00001441    -0.34824012
  H    -0.00000000     0.76001092    -0.93285191
  H     0.00000000    -0.75999650    -0.93290797
  '''
  , basis='6-31g', symmetry=False).build()

h2o_rks    = dft.RKS(h2o)
h2o_rks.xc = "pbe0"

h2o_rks.max_cycle = 100
h2o_rks.conv_tol  = 1e-12
h2o_rks.verbose = 3
h2o_rks.kernel()
dm = h2o_rks.make_rdm1()

rttd = rt.TDSCF(h2o_rks)
rttd.dt = 0.2
rttd.maxstep = 5000
rttd.verbose = 4
prop_scheme = "ep_pc"
rttd.set_prop_func(key="ep_pc")
rttd.efield_vec = lambda t: [delta_efield(t), 0.0, 0.0]
t = time()
rttd.kernel(dm_ao_init=dm)
dip1 = rttd.ndipole
dxx  = rttd.ndipole[:,0] - rttd.ndipole[0,0]

rttd.efield_vec = lambda t: [0.0, delta_efield(t), 0.0]
rttd.kernel(dm_ao_init=dm)
dip2 = rttd.ndipole
dyy  = rttd.ndipole[:,1] - rttd.ndipole[0,1]

rttd.efield_vec = lambda t: [0.0, 0.0, delta_efield(t)]
rttd.kernel(dm_ao_init=dm)
dip3 = rttd.ndipole
dzz  = rttd.ndipole[:,2] - rttd.ndipole[0,2]

dipole_fig, (dipole_axes1, dipole_axes2, dipole_axes3) = plt.subplots(3, 1, sharex=True, figsize=(20,20))
dipole_axes1.set_ylabel('xdipole/a.u.', fontsize=20)
dipole_axes1.set_title('time-dipole, dt=%f, maxstep=%d, prop_method=%s'%(rttd.dt, rttd.maxstep, prop_scheme), fontsize=20)
dipole_axes1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%5.2e'))
dipole_axes1.grid(True)

dipole_axes2.set_ylabel('ydipole/a.u.', fontsize=20)
dipole_axes2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%5.2e'))
dipole_axes2.grid(True)

dipole_axes3.set_xlabel('time/fs', fontsize=20)
dipole_axes3.set_ylabel('zdipole/a.u.', fontsize=20)
dipole_axes3.yaxis.set_major_formatter(ticker.FormatStrFormatter('%5.2e'))
dipole_axes3.grid(True)

dipole_axes1.plot(0.02419*rttd.ntime, dip1[:,0])
dipole_axes2.plot(0.02419*rttd.ntime, dip2[:,1])
dipole_axes3.plot(0.02419*rttd.ntime, dip3[:,2])
dipole_fig.savefig("./h2o_dip.pdf")

lrtd = tddft.TDDFT(h2o_rks)
lrtd.nstates = 30
lrtd.kernel()
mw, sigma = rt.util.build_absorption_spectrum(rttd, ndipole=numpy.array([dxx,dyy,dzz]).T, damp_expo=50.0)

fig, ax = plt.subplots(figsize=(20,10))
ax.plot(27.2116*mw, sigma, label="rt-tddft")
ax.stem(27.2116*lrtd.e, lrtd.oscillator_strength(), linefmt='grey', markerfmt=None, basefmt=" ", use_line_collection=True, label="lr-tddft")
ax.legend()
ax.set_title("the Energy-Absorption plot, dt=%f, maxstep=%d, prop_method=%s"%(rttd.dt, rttd.maxstep, prop_scheme), fontsize=20)

ax.set_xlabel('Energy/eV', fontsize=20)
ax.set_ylabel('Absorption', fontsize=20)
ax.set_xlim(0,25)
ax.set_ylim(0,1)

ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
ax.grid(True)
fig.savefig("./h2o_abos.pdf")
