import pyscf
import numpy

from   pyscf  import dft, scf, tddft
from   pyscf  import gto
from   pyscf  import rt

from pyscf.tools import cubegen
from pyscf.tools import c60struct
from pyscf.rt.result import read_index_list, read_step_dict, read_keyword_value

def apply_field(mol, field=[0,0,0], dm0=None):
    mf    = scf.RHF(mol)
    mf.max_cycle = 100
    h =(mol.intor('cint1e_kin_sph') + mol.intor('cint1e_nuc_sph')
        + numpy.einsum('x,xij->ij', field, mol.intor('cint1e_r_sph', comp=3)))
    mf.get_hcore = lambda *args: h
    mf.conv_tol  = 1e-10
    mf.verbose = 0
    mf.kernel(dm0)
    return mf.make_rdm1()

mol = pyscf.M(atom=[('C', r) for r in c60struct.make60(1.46,1.38)],
                basis='6-31g(d)',
                max_memory=40000)

mf = pyscf.scf.fast_newton(mol.RHF())
dm  = mf.make_rdm1()
dm_ = apply_field(mol, field=[0, 0, 1e-4], dm0=dm)

rttd = rt.TDDFT(mf)
rttd.verbose        = 4
rttd.total_step     = 100
rttd.step_size      = 0.2
rttd.prop_method    = "mmut"
rttd.dm_ao_init     = dm_
rttd.save_in_disk   = True
rttd.chk_file       = "c60.chk"
rttd.sace_in_memory = False
rttd.kernel()

rt_time_list    = read_keyword_value("t",     result_obj=rttd.result_obj)
rt_dm_ao_list   = read_keyword_value("dm_ao", result_obj=rttd.result_obj).real
for i_rt_dm_ao, rt_dm_ao in enumerate(rt_dm_ao_list):
    cubegen.density(mol, 'c60_rt_den_%f.cube'%(rt_time_list[i_rt_dm_ao]), rt_dm_ao-dm)
    
