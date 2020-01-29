# Author: Junjie Yang <yangjunjie0320@gmail.com> Zheng Pei and Yihan Shao

import h5py
from pyscf.lib.chkfile import load_chkfile_key, load
from pyscf.lib.chkfile import dump_chkfile_key, dump, save
from pyscf.lib.chkfile import load_mol, save_mol

def load_rt(chkfile):
    return load_mol(chkfile), load(chkfile, 'rt')

def dump_rt(mol, chkfile, ntime, netot, ndm_ao, overwrite_mol=True):
    '''save temporary results'''
    if h5py.is_hdf5(chkfile) and not overwrite_mol:
        with h5py.File(chkfile, 'a') as fh5:
            if 'mol' not in fh5:
                fh5['mol'] = mol.dumps()
    else:
        save_mol(mol, chkfile)

    rt_dic  = { "ntime"      :ntime,
                "ndm_ao"     :ndm_ao.real,   
                "netot"      :netot}
    save(chkfile, 'rt', rt_dic)

