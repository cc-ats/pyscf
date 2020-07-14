# Author: Junjie Yang <yangjunjie0320@gmail.com> Zheng Pei and Yihan Shao
import time
import numpy
import h5py
from numpy import empty, arange, asarray
from numpy import dot, complex128

from pyscf.lib import StreamObject
from pyscf.lib import logger

# Author: Junjie Yang <yangjunjie0320@gmail.com> Zheng Pei and Yihan Shao
import h5py
from numpy import asarray, sort
from pyscf.lib.chkfile import load_chkfile_key, load
from pyscf.lib.chkfile import dump_chkfile_key, dump, save
from pyscf.lib.chkfile import load_mol, save_mol

def dump_rt_obj(chkfile, rt_obj):
    dump(chkfile, 'rt_result/rt_obj/mol', rt_obj.mol.dumps())

def dump_step_obj(chkfile, **kwargs):
    '''save step results'''
    save(chkfile, 'rt_result/rt_step/%d'%kwargs["save_index"], kwargs)

def read_index_list(result_obj=None, chk_file=None):
    assert (result_obj is not None) or (chk_file is not None)
    if result_obj is not None:
        if result_obj._chk_file is None:
            assert result_obj._time_list is not None
            return result_obj.save_iter_list
        else:
            chk_file = result_obj._chk_file

    if chk_file is not None:
        return load(chk_file, 'rt_result/save_index_list')

def read_step_dict(index, result_obj=None, chk_file=None):
    if result_obj is not None:
        if result_obj._chk_file is None:
            assert result_obj._time_list is not None
            assert result_obj._save_iter_list[index] == index
            temp_step_obj_dict = {
                "save_index":        result_obj._save_iter_list[index],
                "step_index":        result_obj.step_iter_list[index],
                "t":                 result_obj._time_list[index],
                "dm_ao":             result_obj._dm_ao_list[index],
                "dm_orth":           result_obj._dm_orth_list[index],
                "fock_ao":           result_obj._fock_ao_list[index],
                "fock_orth":         result_obj._fock_orth_list[index],
            }
                        
            if (result_obj._dipole_list is not None):
                temp_step_obj_dict["dipole"]   = result_obj._dipole_list[index]
            if (result_obj._pop_list is not None):
                temp_step_obj_dict["pop"]      = result_obj._pop_list[index]
            if (result_obj._energy_elec_list is not None) and (result_obj._energy_tot_list is not None):
                temp_step_obj_dict["energy_elec"] = result_obj._energy_elec_list[index]
                temp_step_obj_dict["energy_tot"]  = result_obj._energy_tot_list[index]
            return temp_step_obj_dict
        else:
            chk_file = result_obj._chk_file

    if chk_file is not None:
        return load(chk_file, 'rt_result/rt_step/%d'%index)

class RealTimeStep(StreamObject):
    def __init__(self, rt_obj, verbose=None):
        if verbose is None:
            verbose = rt_obj.verbose
        self.verbose = verbose
        self.rt_obj = rt_obj
        self.mol    = rt_obj.mol

        self.calculate_dipole = True
        self.calculate_pop    = True
        self.calculate_energy = True

        self.t         = None
        self.step_iter = None

        self.step_energy_elec = None
        self.step_energy_tot  = None
        self.step_dipole      = None
        self.step_pop         = None

        self.step_dm_ao       = None
        self.step_dm_orth     = None
        self.step_fock_orth   = None
        self.step_fock_ao     = None

    def _initialize(self, dm_ao_init, dm_orth_init, fock_ao_init, fock_orth_init,
                         hcore_ao, veff_ao):
        if self.calculate_dipole:
            self.step_dipole = self.rt_obj._scf.dip_moment(dm=dm_ao_init.real, mol=self.mol, verbose=0, unit='au')
        if self.calculate_pop:
            self.step_pop = self.rt_obj._scf.mulliken_pop(dm=dm_ao_init.real, mol=self.mol, verbose=0)

        self.t         = 0.0
        self.step_iter = 0

        self.step_dm_ao         = dm_ao_init
        self.step_dm_orth       = dm_orth_init
        self.step_fock_ao       = fock_ao_init
        self.step_fock_orth     = fock_orth_init

        if self.calculate_energy:
            self.step_energy_elec = self.rt_obj.get_energy_elec(
                hcore_ao, dm_orth=dm_orth_init, dm_ao=dm_ao_init, veff_ao=veff_ao
                )
            self.step_energy_tot = self.rt_obj.get_energy_tot(
                hcore_ao, dm_orth=dm_orth_init, dm_ao=dm_ao_init, veff_ao=veff_ao
                )
        

    def _update(self, t, step_iter, dm_ao, dm_orth, fock_ao, fock_orth, hcore_ao, veff_ao):
        if self.calculate_dipole:
            self.step_dipole = self.rt_obj._scf.dip_moment(dm=dm_ao.real, mol=self.mol, verbose=0, unit='au')
        if self.calculate_pop:
            self.step_pop = self.rt_obj._scf.mulliken_pop(dm=dm_ao.real, mol=self.mol, verbose=0)[1]

        self.t = t
        self.step_iter = step_iter

        self.step_dm_ao     =     dm_ao
        self.step_dm_orth   =   dm_orth
        self.step_fock_ao   =   fock_ao
        self.step_fock_orth = fock_orth

        if self.calculate_energy:
            self.step_energy_elec = self.rt_obj.get_energy_elec(
                hcore_ao, dm_orth=dm_orth, dm_ao=dm_ao, veff_ao=veff_ao
                )
            self.step_energy_tot = self.rt_obj.get_energy_tot(
                hcore_ao, dm_orth=dm_orth, dm_ao=dm_ao, veff_ao=veff_ao
                )

class RealTimeResult(StreamObject):
    def __init__(self, rt_obj, verbose=None):
        if verbose is None:
            verbose = rt_obj.verbose
        self.verbose = verbose 
        self.rt_obj = rt_obj
        self.mol    = rt_obj.mol

        self.save_iter          = None
        self.save_iter_list     = None

        self._time_list         = None
        self.step_iter_list    = None

        self._energy_elec_list = None
        self._energy_tot_list  = None
        self._dipole_list      = None
        self._pop_list         = None

        self._dm_ao_list       = None
        self._dm_orth_list     = None
        self._fock_orth_list   = None
        self._fock_ao_list     = None

        self._save_in_disk       = False
        self._chk_file           = None
        self._save_in_memory     = True

    def _initialize(self, first_step_obj):
        assert self._save_in_disk or self._save_in_memory
        if self._save_in_memory:
            logger.info(self, "The results would be saved in the memory.")
            cput_time = (time.clock(), time.time())
            if (first_step_obj.step_dipole is not None):
                self._dipole_list   = [first_step_obj.step_dipole]
            if (first_step_obj.step_pop is not None):
                self._pop_list      = [first_step_obj.step_pop[1]]
            if (first_step_obj.step_energy_elec is not None) and (first_step_obj.step_energy_tot is not None):
                self._energy_elec_list = [first_step_obj.step_energy_elec]
                self._energy_tot_list  = [first_step_obj.step_energy_tot ]

            self._time_list        = [first_step_obj.t               ]
            self._dm_ao_list       = [first_step_obj.step_dm_ao      ]
            self._dm_orth_list     = [first_step_obj.step_dm_orth    ]
            self._fock_ao_list     = [first_step_obj.step_fock_ao    ]
            self._fock_orth_list   = [first_step_obj.step_fock_orth  ]
            cput1 = logger.timer(self, 'The first step is saved in the memory:', *cput_time)

        if self._save_in_disk and self._chk_file is not None:
            logger.info(self, "The results would be saved in the disk, %s.", self._chk_file)
            cput_time = (time.clock(), time.time())
            temp_step_obj_dict = {
                "save_index":        0,
                "step_index":        0,
                "t":                 0.0,
                "dm_ao":             first_step_obj.step_dm_ao,
                "dm_orth":           first_step_obj.step_dm_orth,  
                "fock_orth":         first_step_obj.step_fock_orth,
                "fock_ao":           first_step_obj.step_fock_ao,  
            }
            if (first_step_obj.step_dipole is not None):
                temp_step_obj_dict["dipole"]   = first_step_obj.step_dipole
            if (first_step_obj.step_pop is not None):
                temp_step_obj_dict["pop"]      = first_step_obj.step_pop
            if (first_step_obj.step_energy_elec is not None) and (first_step_obj.step_energy_tot is not None):
                temp_step_obj_dict["energy_elec"] = first_step_obj.step_energy_elec
                temp_step_obj_dict["energy_tot"]  = first_step_obj.step_energy_tot
            dump_step_obj(self._chk_file, **temp_step_obj_dict)
            cput1 = logger.timer(self, 'The first step is saved in the disk:', *cput_time)

        self.save_iter_list   = [0]
        self.save_iter        = 0
        self.step_iter_list  = [0]

        return 0

    def _update(self, step_obj):
        if self._save_in_memory:
            cput_time = (time.clock(), time.time())
            if (step_obj.step_dipole is not None):
                self._dipole_list.append(step_obj.step_dipole)
            if (step_obj.step_pop is not None):
                self._pop_list.append(step_obj.step_pop)
            if (step_obj.step_energy_elec is not None) and (step_obj.step_energy_tot is not None):
                self._energy_elec_list.append( step_obj.step_energy_elec)
                self._energy_tot_list.append(  step_obj.step_energy_tot )

            self._time_list.append(        step_obj.t               )
            self._dm_ao_list.append(       step_obj.step_dm_ao      )
            self._dm_orth_list.append(     step_obj.step_dm_orth    )
            self._fock_orth_list.append(   step_obj.step_fock_orth  )
            self._fock_ao_list.append(     step_obj.step_fock_ao    )
            cput1 = logger.timer(self, 'The %d step is saved in the memory:'%step_obj.step_iter, *cput_time)

        if self._save_in_disk and self._chk_file is not None:
            cput_time = (time.clock(), time.time())
            temp_step_obj_dict = {
                "save_index":        self.save_iter+1,
                "step_index":        step_obj.step_iter,
                "t":                 step_obj.t,
                "dm_ao":             step_obj.step_dm_ao,
                "dm_orth":           step_obj.step_dm_orth,  
                "fock_orth":         step_obj.step_fock_orth,
                "fock_ao":           step_obj.step_fock_ao,  
            }
            if (step_obj.step_dipole is not None):
                temp_step_obj_dict["dipole"]   = step_obj.step_dipole
            if (step_obj.step_pop is not None):
                temp_step_obj_dict["pop"]   = step_obj.step_pop
            if (step_obj.step_energy_elec is not None) and (step_obj.step_energy_tot is not None):
                temp_step_obj_dict["energy_elec"] = step_obj.step_energy_elec
                temp_step_obj_dict["energy_tot"]  = step_obj.step_energy_tot
            dump_step_obj(self._chk_file, **temp_step_obj_dict)
            cput1 = logger.timer(self, 'The %d step is saved in the disk:  '%step_obj.step_iter, *cput_time)

        self.save_iter = self.save_iter + 1
        self.save_iter_list.append(self.save_iter)
        self.step_iter_list.append(step_obj.step_iter)

        return self.save_iter

    def _finalize(self):
        if self._save_in_disk and self._chk_file is not None:
            logger.info(self, '%d out of %d steps are saved in the memory.'%(len(self.save_iter_list), self.step_iter_list[-1]+1))
        if self._save_in_disk and self._chk_file is not None:
            logger.info(self, '%d out of %d steps are saved in the disk.  '%(len(self.save_iter_list), self.step_iter_list[-1]+1))
            save(self._chk_file, 'rt_result/save_index_list', asarray(self.save_iter_list, dtype=int))