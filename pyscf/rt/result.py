# Author: Junjie Yang <yangjunjie0320@gmail.com> Zheng Pei and Yihan Shao
import numpy
from numpy import empty, arange
from numpy import dot, complex128

from pyscf.rt.chkfile import load_rt_step_index, load_rt_step
from pyscf.rt.chkfile import dump_rt_obj, dump_rt_step
from pyscf.lib import StreamObject
from pyscf.lib import logger

class RealTimeStep(StreamObject):
    def __init__(self, rt_obj, verbose=None):
        if verbose is None:
            verbose = rt_obj.verbose
        self.verbose = verbose
        self.rt_obj = rt_obj
        self.mol    = rt_obj.mol

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
                         hcore_ao, veff_ao,
                         calculate_dipole=None, calculate_pop=None, calculate_energy=None):
        if (calculate_dipole is None):
            calculate_dipole = self.rt_obj.calculate_dipole
        if calculate_dipole:
            self.step_dipole = self.rt_obj._scf.dip_moment(dm=dm_ao_init.real, mol=self.mol, verbose=0, unit='au')
        if calculate_pop is None:
            calculate_pop = self.rt_obj.calculate_pop
        if calculate_pop:
            self.step_pop = self.rt_obj._scf.mulliken_pop(dm=dm_ao_init.real, mol=self.mol, verbose=0)

        self.t         = 0.0
        self.step_iter = 0

        self.step_dm_ao         = dm_ao_init
        self.step_dm_orth       = dm_orth_init
        self.step_fock_ao       = fock_ao_init
        self.step_fock_orth     = fock_orth_init

        logger.debug(self, 'step_iter=%d, t=%f', 0, 0.0)
        if calculate_energy is None:
            calculate_energy = self.rt_obj.calculate_energy
        if calculate_energy:
            self.step_energy_elec = self.rt_obj.get_energy_elec(
                hcore_ao, dm_orth=dm_orth_init, dm_ao=dm_ao_init, veff_ao=veff_ao
                )
            self.step_energy_tot = self.rt_obj.get_energy_tot(
                hcore_ao, dm_orth=dm_orth_init, dm_ao=dm_ao_init, veff_ao=veff_ao
                )
            logger.debug(self, 'step_energy_elec = %f, step_energy_tot = %f', self.step_energy_elec, self.step_energy_tot)
        

    def _update(self, t, step_iter, dm_ao, dm_orth, fock_ao, fock_orth, hcore_ao, veff_ao):
        if (self.step_dipole is not None):
            self.step_dipole = self.rt_obj._scf.dip_moment(dm=dm_ao.real, mol=self.mol, verbose=0, unit='au')
        if (self.step_pop is not None):
            self.step_pop = self.rt_obj._scf.mulliken_pop(dm=dm_ao.real, mol=self.mol, verbose=0)[1]

        self.t = t
        self.step_iter = step_iter

        self.step_dm_ao     =     dm_ao
        self.step_dm_orth   =   dm_orth
        self.step_fock_ao   =   fock_ao
        self.step_fock_orth = fock_orth

        logger.debug(self, 'step_iter=%d, t=%f', step_iter, t)
        if (self.step_energy_elec is not None) and (self.step_energy_tot is not None):
            self.step_energy_elec = self.rt_obj.get_energy_elec(
                hcore_ao, dm_orth=dm_orth, dm_ao=dm_ao, veff_ao=veff_ao
                )
            self.step_energy_tot = self.rt_obj.get_energy_tot(
                hcore_ao, dm_orth=dm_orth, dm_ao=dm_ao, veff_ao=veff_ao
                )
            logger.debug(self, 'step_energy_elec = %f, step_energy_tot = %f', self.step_energy_elec, self.step_energy_tot)

class RealTimeResult(object):
    def __init__(self, rt_obj, verbose=None):
        if verbose is None:
            verbose = rt_obj.verbose
        self.verbose = verbose 
        self.rt_obj = rt_obj
        self.mol    = rt_obj.mol

        self.chk_file           = None

        self.save_iter          = None
        self.save_iter_list     = None

        self._time_list         = None
        self._step_iter_list    = None
        self._save_iter_list    = None

        self._energy_elec_list = None
        self._energy_tot_list  = None
        self._dipole_list      = None
        self._pop_list         = None

        self._dm_ao_list       = None
        self._dm_orth_list     = None
        self._fock_orth_list   = None
        self._fock_ao_list     = None

    def _initialize(self, first_step_obj):
        if (first_step_obj.step_dipole is not None):
            self._dipole_list   = [first_step_obj.step_dipole]
        if (first_step_obj.step_pop is not None):
            self._pop_list      = [first_step_obj.step_pop[1]]
        if (first_step_obj.step_energy_elec is not None) and (first_step_obj.step_energy_tot is not None):
            self._energy_elec_list = [first_step_obj.step_energy_elec]
            self._energy_tot_list  = [first_step_obj.step_energy_tot ]

        self._save_iter_list   = [0]

        self._time_list        = [first_step_obj.t               ]
        self._step_iter_list   = [first_step_obj.step_iter       ]

        self._dm_ao_list       = [first_step_obj.step_dm_ao      ]
        self._dm_orth_list     = [first_step_obj.step_dm_orth    ]
        self._fock_ao_list     = [first_step_obj.step_fock_ao    ]
        self._fock_orth_list   = [first_step_obj.step_fock_orth  ]

        self.save_iter = 0
        return 0

    def _update(self, step_obj):
        if (step_obj.step_dipole is not None):
            self._dipole_list.append(step_obj.step_dipole)
        if (step_obj.step_pop is not None):
            self._pop_list.append(step_obj.step_pop)
        if (step_obj.step_energy_elec is not None) and (step_obj.step_energy_tot is not None):
            self._energy_elec_list.append( step_obj.step_energy_elec)
            self._energy_tot_list.append(  step_obj.step_energy_tot )

        self._time_list.append(        step_obj.t               )
        self._step_iter_list.append(   step_obj.step_iter       )
        self._dm_ao_list.append(       step_obj.step_dm_ao      )
        self._dm_orth_list.append(     step_obj.step_dm_orth    )
        self._fock_orth_list.append(   step_obj.step_fock_orth  )
        self._fock_ao_list.append(     step_obj.step_fock_ao    )

        self.save_iter = self.save_iter + 1
        self._save_iter_list.append(self.save_iter)

        return self.save_iter

    def _finalize(self):
        pass

    def get_step_dict(self, step_iter=None, save_iter=None, t=None):
        pass


    # def load_rt_step_index(self):
    #     return load_rt_step_index(self.chkfile)

    # def load_rt_step(self, step_index):
    #     if hasattr(step_index, '__iter__'):
    #         return [load_rt_step(self.chkfile, istep_index) for istep_index in step_index]
    #     else:
    #         return load_rt_step(self.chkfile, step_index)

def save_step(step_obj, result_obj = None, save_in_memory = None,
                   chk_file = None,  save_in_disk = None):
    pass