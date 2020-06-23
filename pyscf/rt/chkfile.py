# Author: Junjie Yang <yangjunjie0320@gmail.com> Zheng Pei and Yihan Shao

import h5py
from pyscf.lib.chkfile import load_chkfile_key, load
from pyscf.lib.chkfile import dump_chkfile_key, dump, save
from pyscf.lib.chkfile import load_mol, save_mol

# def load_rt_obj(chkfile):
#     '''save step results'''
#     load(chkfile, 'rt_obj/mol', rt_obj.mol.dumps())

def load_rt_step_index(chkfile):
    with h5py.File(chkfile, 'r') as hf:
        step_index = list(hf['rt_step'].keys())
    return range(len(step_index))

def load_rt_step(chkfile, index):
    return load(chkfile, 'rt_step/%s'%index)

def dump_rt_obj(chkfile, rt_obj):
    '''save step results'''
    dump(chkfile, 'rt_obj/mol', rt_obj.mol.dumps())

def dump_rt_step(chkfile, idx, **kwargs):
    '''save step results'''
    rt_dic  =   kwargs
    save(chkfile, 'rt_step/%d'%idx, rt_dic)

if __name__ == "__main__":
    from field import ClassicalElectricField, gaussian_field_vec
    import time
    import tempfile
    import numpy

    from pyscf import gto, scf
    from pyscf import lib, lo
    from pyscf.rt.rhf  import TDHF
    from pyscf.rt.rhf  import orth_canonical_mo, ao2orth_contravariant, ao2orth_covariant
    from pyscf.rt.util import print_matrix, print_cx_matrix

    from field import ClassicalElectricField, gaussian_field_vec
    h2o =   gto.Mole( atom='''
    O     0.00000000    -0.00001441    -0.34824012
    H    -0.00000000     0.76001092    -0.93285191
    H     0.00000000    -0.75999650    -0.93290797
    '''
    , basis='sto-3g', symmetry=False).build()

    h2o_rhf    = scf.RHF(h2o)
    h2o_rhf.verbose = 4
    h2o_rhf.conv_tol = 1e-12
    h2o_rhf.kernel()

    dm = h2o_rhf.make_rdm1()
    fock = h2o_rhf.get_fock()

    orth_xtuple = orth_canonical_mo(h2o_rhf)
    dm_orth = ao2orth_contravariant(dm, orth_xtuple)
    fock_orth_0 = ao2orth_covariant(fock, orth_xtuple)
    
    gau_vec = lambda t: gaussian_field_vec(t, 1.0, 1.0, 0.0, [0.020,0.00,0.00])
    gaussian_field = ClassicalElectricField(h2o, field_func=gau_vec, stop_time=10.0)

    rttd = TDHF(h2o_rhf, field=gaussian_field)
    rttd.verbose = 4
    rttd.maxstep = 10
    rttd.dt      = 0.02
    rttd._initialize()
    
    h1e = rttd.get_hcore_ao(5.0)
    veff_ao = rttd.get_veff_ao(dm_orth, dm_ao=dm)
    fock_orth = rttd.get_fock_orth(h1e, dm_orth, dm_ao=dm, veff_ao=veff_ao)

    _chkfile    = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    tmp_chkfile = _chkfile.name

    dump_rt_obj(tmp_chkfile, rttd)
    for it, t in enumerate(numpy.linspace(0,100,1001)):
        dump_rt_step(tmp_chkfile, it,
        t=t, etot=10.00, dm_ao=dm, dm_orth=dm_orth, fock_ao=fock, fock_orth=fock_orth)

    step_index = load_rt_step_index(tmp_chkfile)

    for step in step_index:
        print("step_index = ", step)
        rtstep = load_rt_step(tmp_chkfile, step)
        print("t = %f"%rtstep["t"])
    