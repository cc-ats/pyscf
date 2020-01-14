import numpy
import scipy
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

from functools import reduce

import pyscf
from   pyscf import dft, scf
from   pyscf import gto
from   pyscf  import rt

def delta_efield(t):
    return 0.0001*numpy.exp(-10*t**2/0.2**2)

o2 =   gto.Mole( atom='''
  O    0.0000000    0.0000000    0.5754646
  O    0.0000000    0.0000000   -0.5754646
  '''
  , basis='6-31g', spin=2, symmetry=False).build()

o2_uks    = dft.UKS(o2)
o2_uks.xc = "pbe0"

o2_uks.max_cycle = 100
o2_uks.conv_tol  = 1e-12
o2_uks.verbose = 3
o2_uks.kernel()
dm = o2_uks.make_rdm1()

rttd = rt.TDSCF(o2_uks)
rttd.dt = 0.2
rttd.maxstep = 1000
rttd.verbose = 5
for prop_scheme in ["euler", "mmut", "amut1", "amut2", "amut3", "ep_pc", "lflp_pc"]:
    rttd.set_prop_func(key=prop_scheme)
    rttd.efield_vec = lambda t: [delta_efield(t), 0.0, 0.0]
    rttd.chkfile = 'o2_rt_x.chk'
    rttd.kernel(dm_ao_init=dm)

    rttd.efield_vec = lambda t: [0.0, delta_efield(t), 0.0]
    rttd.chkfile = 'o2_rt_y.chk'
    rttd.kernel(dm_ao_init=dm)

    rttd.efield_vec = lambda t: [0.0, 0.0, delta_efield(t)]
    rttd.chkfile = 'o2_rt_z.chk'
    rttd.kernel(dm_ao_init=dm)

    dip1 = numpy.zeros([rttd.maxstep + 1, 3])
    dip2 = numpy.zeros([rttd.maxstep + 1, 3])
    dip3 = numpy.zeros([rttd.maxstep + 1, 3])

    rttd.__dict__.update(scf.chkfile.load('./o2_rt_x.chk', 'rt'))
    for i,idm in enumerate(rttd.ndm_ao):
        dip1[i,:] = rttd.mf.dip_moment(dm = idm.real, unit='au', verbose=0)
    dxx  = dip1[:,0] - dip1[0,0]
    ene1 = numpy.array(rttd.netot[1:] - rttd.netot[1])

    rttd.__dict__.update(scf.chkfile.load('./o2_rt_y.chk', 'rt'))
    for i,idm in enumerate(rttd.ndm_ao):
        dip2[i,:] = rttd.mf.dip_moment(dm = idm.real, unit='au', verbose=0)
    dyy = dip2[:,1] - dip2[0,1]
    ene2 = numpy.array(rttd.netot[1:] - rttd.netot[1])

    rttd.__dict__.update(scf.chkfile.load('./o2_rt_z.chk', 'rt'))
    for i,idm in enumerate(rttd.ndm_ao):
        dip3[i,:] = rttd.mf.dip_moment(dm = idm.real, unit='au', verbose=0)
    dzz = dip3[:,2] - dip3[0,2]
    ene3 = numpy.array(rttd.netot[1:] - rttd.netot[1])

    dipole_fig, (dipole_axes1, dipole_axes2, dipole_axes3) = plt.subplots(3, 1, sharex=True, figsize=(20,20))
    dipole_axes1.set_ylabel('xdipole/a.u.')
    dipole_axes1.set_title('time-dipole, dt=%f, maxstep=%d, prop_method=%s'%(rttd.dt, rttd.maxstep, prop_scheme))
    dipole_axes1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%5.2e'))
    dipole_axes1.grid(True)

    dipole_axes2.set_ylabel('ydipole/a.u.')
    dipole_axes2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%5.2e'))
    dipole_axes2.grid(True)

    dipole_axes3.set_xlabel('time/fs')
    dipole_axes3.set_ylabel('zdipole/a.u.')
    dipole_axes3.yaxis.set_major_formatter(ticker.FormatStrFormatter('%5.2e'))
    dipole_axes3.grid(True)

    dipole_axes1.plot(0.02419*rttd.ntime, dip1[:,0])
    dipole_axes2.plot(0.02419*rttd.ntime, dip2[:,1])
    dipole_axes3.plot(0.02419*rttd.ntime, dip3[:,2])
    dipole_fig.savefig("./o2_dipole_fig_%s.pdf"%prop_scheme)

    ene_fig, (ene_axes1, ene_axes2, ene_axes3) = plt.subplots(3, 1, sharex=True, figsize=(20,20))
    ene_axes1.set_ylabel('ene err/a.u.')
    ene_axes1.set_title('time-energy, dt=%f, maxstep=%d, prop_method=%s'%(rttd.dt, rttd.maxstep, prop_scheme))
    ene_axes1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%5.2e'))
    ene_axes1.grid(True)

    ene_axes2.set_ylabel('ene err/a.u.')
    ene_axes2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%5.2e'))
    ene_axes2.grid(True)

    ene_axes3.set_xlabel('time/fs')
    ene_axes3.set_ylabel('ene err/a.u.')
    ene_axes3.yaxis.set_major_formatter(ticker.FormatStrFormatter('%5.2e'))
    ene_axes3.grid(True)

    ene_axes1.plot(0.02419*rttd.ntime[1:], ene1)
    ene_axes2.plot(0.02419*rttd.ntime[1:], ene2)
    ene_axes3.plot(0.02419*rttd.ntime[1:], ene3)
    ene_fig.savefig("./o2_ene_fig_%s.pdf"%prop_scheme)
