#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#
import h5py
from pyscf.lib.chkfile import load_chkfile_key, load
from pyscf.lib.chkfile import dump_chkfile_key, dump, save
from pyscf.lib.chkfile import load_mol, save_mol

def load_rt(chkfile):
    return load_mol(chkfile), load(chkfile, 'rt')

def dump_rt(mol, chkfile, ntime, ndm_prim, ndm_ao,
            nfock_prim, nfock_ao, netot, overwrite_mol=True):
    '''save temporary results'''
    if h5py.is_hdf5(chkfile) and not overwrite_mol:
        with h5py.File(chkfile, 'a') as fh5:
            if 'mol' not in fh5:
                fh5['mol'] = mol.dumps()
    else:
        save_mol(mol, chkfile)

    rt_dic  = { "ntime"      :ntime,      
                "ndm_prim"   :ndm_prim,   
                "ndm_ao"     :ndm_ao,     
                "nfock_prim" :nfock_prim, 
                "nfock_ao"   :nfock_ao,   
                "netot"      :netot}
    save(chkfile, 'rt', rt_dic)

