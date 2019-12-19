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
h2o_rks.xc = "pbe"
h2o_rks.max_cycle = 100
h2o_rks.conv_tol = 1e-20
h2o_rks.verbose = 5
h2o_rks.kernel()

def delta_efield(tt):
    return 0.0001*np.exp(-10*tt**2/0.2**2)

rttd.efield_vec = lambda t: [delta_efield(t), 0.0, 0.0]
rttd.kernel(dm_ao_init=h2o_rks.make_rdm1())
dxx = rttd.ndipole[:,0] - rttd.ndipole[0,0]

rttd.efield_vec = lambda t: [0.0, delta_efield(t), 0.0]
rttd.kernel(dm_ao_init=h2o_rks.make_rdm1())
dyy = rttd.ndipole[:,1] - rttd.ndipole[0,1]

rttd.efield_vec = lambda t: [0.0, 0.0, delta_efield(t)]
rttd.kernel(dm_ao_init=h2o_rks.make_rdm1())
dzz = rttd.ndipole[:,2] - rttd.ndipole[0,2]

#%%
dipole_fig, (dipole_axes1, dipole_axes2, dipole_axes3) = plt.subplots(3, 1, sharex=True, figsize=(20,20))
dipole_axes1.set_ylabel('xdipole/a.u.')
dipole_axes1.set_title('time-dipole, dt=%f, maxstep=%d, prop_method=%s'%(rttd.dt, rttd.maxstep, prop_method))
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
dipole_fig.savefig("./testlog/dipole_fig_%s.pdf"%prop_method)

#%%
lrtd = tddft.TDDFT(h2o_rks)
lrtd.nstates = 30
lrtd.kernel()
mw, sigma = rt.build_absorption_spectrum(rttd, ndipole=np.array([dxx,dyy,dzz]).T, damp_expo=200)

fig, ax = plt.subplots(figsize=(20,10))
ax.plot(27.2116*mw, sigma, label="rt-tddft")
ax.stem(27.2116*lrtd.e, lrtd.oscillator_strength(), linefmt='grey', markerfmt=None, basefmt=" ", use_line_collection=True, label="lr-tddft")
ax.legend()
ax.set_title("the Energy-Absorption plot, dt=%f, maxstep=%d, prop_method=%s"%(rttd.dt, rttd.maxstep, prop_method))

ax.set_xlabel('Energy/eV')
ax.set_xlim(0,60)
ax.set_ylim(0,1)

ax.set_ylabel('Absorption')
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
ax.grid(True)
fig.savefig("./testlog/abos_fig_%s.pdf"%prop_method)


rttd = rt.TDSCF(ag4_n2_rks1)
rttd.__dict__.update(scf.chkfile.load('./chk/rttd.chk', 'rt'))