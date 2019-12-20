import numpy as np
import pyscf
import pyscf.dft
import pyscf.gto
from pyscf import rt

#%%

from pyscf import gto, scf, dft, tddft

import scipy
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

plt.style.use("default")

h2o =   gto.Mole( atom='''
  H   -0.0000000    0.4981795    0.7677845
  O   -0.0000000   -0.0157599    0.0000000
  H    0.0000000    0.4981795   -0.7677845
  '''
  , basis='6-31g(d)', symmetry=False).build()

h2o_rks = dft.RKS(h2o)
h2o_rks.xc = "pbe0"
h2o_rks.max_cycle = 100
h2o_rks.conv_tol = 1e-20
h2o_rks.verbose = 5
h2o_rks.kernel()

rttd = rt.TDSCF(h2o_rks)
rttd.__dict__.update(scf.chkfile.load('./h2o_rt_x.chk', 'rt'))

maxstep = rttd.nstep.size
dt      = rttd.ntime[1] - rttd.ntime[0]

rttd.maxstep = maxstep
rttd.dt      = dt

dip1 = np.zeros([maxstep, 3])
dip2 = np.zeros([maxstep, 3])
dip3 = np.zeros([maxstep, 3])

for i,idm in enumerate(rttd.ndm_ao):
    dip1[i,:] = rttd.mf.dip_moment(dm = idm.real, unit='au', verbose=0)
dxx = dip1[:,1] - dip1[0,1]

rttd.__dict__.update(scf.chkfile.load('./h2o_rt_y.chk', 'rt'))
for i,idm in enumerate(rttd.ndm_ao):
    dip2[i,:] = rttd.mf.dip_moment(dm = idm.real, unit='au', verbose=0)
dyy = dip2[:,1] - dip2[0,1]

rttd.__dict__.update(scf.chkfile.load('./h2o_rt_z.chk', 'rt'))
for i,idm in enumerate(rttd.ndm_ao):
    dip3[i,:] = rttd.mf.dip_moment(dm = idm.real, unit='au', verbose=0)
dzz = dip3[:,1] - dip3[0,1]

#%%
dipole_fig, (dipole_axes1, dipole_axes2, dipole_axes3) = plt.subplots(3, 1, sharex=True, figsize=(20,20))
dipole_axes1.set_ylabel('xdipole/a.u.')
dipole_axes1.set_title('time-dipole, dt=%f, maxstep=%d, prop_method=%s'%(dt, maxstep, "AMUT3"))
dipole_axes1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
dipole_axes1.grid(True)

dipole_axes2.set_ylabel('ydipole/a.u.')
# dipole_axes2.set_ylim(-1e-5, 8e-5)
dipole_axes2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
dipole_axes2.grid(True)

dipole_axes3.set_xlabel('time/fs')
dipole_axes3.set_ylabel('zdipole/a.u.')
dipole_axes3.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
dipole_axes3.grid(True)

ntime = rttd.ntime
dipole_axes1.plot(0.02419*ntime, dxx)
dipole_axes2.plot(0.02419*ntime, dyy)
dipole_axes3.plot(0.02419*ntime, dzz)
dipole_fig.savefig("./dipole_fig_%s.pdf"%"AMUT3")

#%%
lrtd = tddft.TDDFT(h2o_rks)
lrtd.nstates = 30
lrtd.kernel()

mw, sigma = rt.build_absorption_spectrum(rttd, ndipole=np.array([dxx,dyy,dzz]).T, damp_expo=500)

fig, ax = plt.subplots(figsize=(20,10))
ax.plot(27.2116*mw, sigma, label="rt-tddft")
ax.stem(27.2116*lrtd.e, lrtd.oscillator_strength(), linefmt='grey', markerfmt=None, basefmt=" ", use_line_collection=True, label="lr-tddft")
ax.legend()
ax.set_title("the Energy-Absorption plot, dt=%f, maxstep=%d, prop_method=%s"%(dt, maxstep, "AMUT3"))

ax.set_xlabel('Energy/eV')
ax.set_xlim(0,100)
ax.set_ylim(0,1)

ax.set_ylabel('Absorption')
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
ax.grid(True)
fig.savefig("./abos_fig_%s.pdf"%"amut3")