# Author: Junjie Yang <yangjunjie0320@gmail.com> Zheng Pei and Yihan Shao

import h5py
from pyscf.lib.chkfile import load_chkfile_key, load
from pyscf.lib.chkfile import dump_chkfile_key, dump, save
from pyscf.lib.chkfile import load_mol, save_mol

def load_rt(chkfile):
    return load_mol(chkfile), load(chkfile, 'rt')

def load_rt_step(chkfile, t):
    return load(chkfile, 'rt_step_%f'%t)

def dump_rt(mol, chkfile, ntime, netot, ndm_ao, overwrite_mol=True):
    '''save temporary results'''
    if h5py.is_hdf5(chkfile) and not overwrite_mol:
        with h5py.File(chkfile, 'a') as fh5:
            if 'mol' not in fh5:
                fh5['mol'] = mol.dumps()
    else:
        save_mol(mol, chkfile)

    rt_dic  = { "ntime"      :ntime,
                "ndm_ao"     :ndm_ao,   
                "netot"      :netot}
    save(chkfile, 'rt', rt_dic)


def dump_step(chkfile, t, etot, dm_ao, dm_orth, fock_ao, fock_orth):
    '''save step results'''

    rt_dic  =   {"t"         :t,
                "etot"       :etot,   
                "dm_ao"      :dm_ao,
                "dm_orth"    :dm_orth,
                "fock_ao"    :fock_ao,   
                "fock_orth"  :fock_orth}
    save(chkfile, 'rt_step_%f'%t, rt_dic)

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
    print_cx_matrix("dm_orth = ", dm_orth)
    fock_orth_0 = ao2orth_covariant(fock, orth_xtuple)
    print_cx_matrix("fock_orth_0 = ", fock_orth_0)
    
    gau_vec = lambda t: gaussian_field_vec(t, 1.0, 1.0, 0.0, [0.020,0.00,0.00])
    gaussian_field = ClassicalElectricField(h2o, field_func=gau_vec, stop_time=10.0)

    rttd = TDHF(h2o_rhf, field=gaussian_field)
    rttd.verbose = 4
    rttd._initialize()

    h1e = rttd.get_hcore_ao(5.0)
    veff_ao = rttd.get_veff_ao(dm_orth, dm_ao=dm)
    fock_orth = rttd.get_fock_orth(h1e, dm_orth, dm_ao=dm, veff_ao=veff_ao)
    print_cx_matrix("fock_orth - fock_orth_0 = ", fock_orth-fock_orth_0)

    h1e = rttd.get_hcore_ao(12.0)
    veff_ao = rttd.get_veff_ao(dm_orth, dm_ao=dm)
    fock_orth = rttd.get_fock_orth(h1e, dm_orth, dm_ao=dm, veff_ao=veff_ao)
    print_cx_matrix("fock_orth - fock_orth_0 = ", fock_orth-fock_orth_0)

    _chkfile    = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    tmp_chkfile = _chkfile.name
    dump_step(tmp_chkfile, 5.0, 10.00, dm, dm_orth, fock, fock_orth)
    dump_step(tmp_chkfile, 10.0, 10.00, dm, dm_orth, fock, fock_orth)

    ans_list = load_rt_step(tmp_chkfile, 5.00)
    print(ans_list)

    ans_list = load_rt_step(tmp_chkfile, 10.00)
    print(ans_list)

    save(tmp_chkfile, 'haha', {"1":1})
    save(tmp_chkfile, 'hehe', {"1":1, "2":2})
    with h5py.File(tmp_chkfile, 'r') as hf:
        print(hf["haha"])