# Author: Junjie Yang <yangjunjie0320@gmail.com> Zheng Pei and Yihan Shao
import numpy
from numpy import empty, arange
from numpy import dot, complex128


class RealTimeStep(object):
    def __init__(self, rt_obj):
        self.rt_obj = rt_obj
        self.mol    = rt_obj.mol

        self.t         = None
        self.iter_step = None

        self.step_energy_elec = None
        self.step_energy_tot  = None
        self.step_dipole      = None
        self.step_pop         = None

        self.step_dm_ao       = None
        self.step_dm_orth     = None
        self.step_fock_orth   = None
        self.step_fock_ao     = None

    def initialize(self, dm_ao_init, dm_orth_init, fock_ao_init, fock_orth_init,
                         hcore_ao, veff_ao, calculate_dipole=None, calculate_pop=None):
        if (calculate_dipole is None):
            calculate_dipole = self.rt_obj.calculate_dipole
        if calculate_dipole:
            self.step_dipole = self.rt_obj._scf.dip_moment(dm=dm_ao_init.real, mol=self.mol, verbose=0, unit='au')
        if calculate_pop is None:
            calculate_pop = self.rt_obj.calculate_pop
        if calculate_pop:
            self.step_pop = self.rt_obj._scf.mulliken_pop(dm=dm_ao_init.real, mol=self.mol, verbose=0)

        self.t     = 0.0
        self.iter_step = 0

        self.step_dm_ao         = dm_ao_init
        self.step_dm_orth       = dm_orth_init
        self.step_fock_ao       = fock_ao_init
        self.step_fock_orth     = fock_orth_init

        self.step_energy_elec = self.rt_obj.get_energy_elec(
            hcore_ao, dm_orth=dm_orth_init, dm_ao=dm_ao_init, veff_ao=veff_ao
            )
        self.step_energy_tot = self.rt_obj.get_energy_tot(
            hcore_ao, dm_orth=dm_orth_init, dm_ao=dm_ao_init, veff_ao=veff_ao
            )

    def update_step(self, t, iter_step, dm_ao, dm_orth, fock_ao, fock_orth, hcore_ao, veff_ao):
        if (self.step_dipole is not None):
            self.step_dipole = self.rt_obj._scf.dip_moment(dm=dm_ao.real, mol=self.mol, verbose=0, unit='au')
        if (self.step_pop is not None):
            self.step_pop = self.rt_obj._scf.mulliken_pop(dm=dm_ao.real, mol=self.mol, verbose=0)

        self.t = t
        self.iter_step = iter_step

        self.step_dm_ao     =     dm_ao
        self.step_dm_orth   =   dm_orth
        self.step_fock_ao   =   fock_ao
        self.step_fock_orth = fock_orth

        self.step_energy_elec = self.rt_obj.get_energy_elec(
            hcore_ao, dm_orth=dm_orth, dm_ao=dm_ao, veff_ao=veff_ao
            )
        self.step_energy_tot = self.rt_obj.get_energy_tot(
            hcore_ao, dm_orth=dm_orth, dm_ao=dm_ao, veff_ao=veff_ao
            )

    def save_step_to_result(self):
        pass

    def save_step_to_chk(self):
        pass


class RealTimeResult(object):
    def __init__(self, rt_obj):
        self.rt_obj = rt_obj
        self.mol    = rt_obj.mol

        self._time_list     = None
        self._iter_step_list    = None

        self._energy_elec_list = None
        self._energy_tot_list  = None
        self._dipole_list      = None
        self._pop_list         = None

        self._dm_ao_list       = None
        self._dm_orth_list     = None
        self._fock_orth_list   = None
        self._fock_ao_list     = None

    def initialize(self, first_rt_step):
        if (first_rt_step.step_dipole is not None):
            self._dipole_list   = [first_rt_step.step_dipole]
        if (first_rt_step.step_pop is not None):
            self._pop_list      = [first_rt_step.step_pop]

        matrix_shape = list(first_rt_step.step_dm_ao.shape)

        self._time_list        = [first_rt_step.t               ]
        self._iter_step_list       = [first_rt_step.iter_step           ]
        self._energy_elec_list = [first_rt_step.step_energy_elec]
        self._energy_tot_list  = [first_rt_step.step_energy_tot ]
        self._dm_ao_list       = [first_rt_step.step_dm_ao      ]
        self._dm_orth_list     = [first_rt_step.step_dm_orth    ]
        self._fock_orth_list   = [first_rt_step.step_fock_ao    ]
        self._fock_ao_list     = [first_rt_step.step_fock_orth  ]

    def update_step(self, rt_step):
        if (rt_step.step_dipole is not None):
            self._dipole_list.append(rt_step.step_dipole)
        if (rt_step.step_pop is not None):
            self._pop_list.append(rt_step.step_pop)

        self._time_list.append(        rt_step.t               )
        self._iter_step_list.append(   rt_step.iter_step       )
        self._energy_elec_list.append( rt_step.step_energy_elec)
        self._energy_tot_list.append(  rt_step.step_energy_tot )
        self._dm_ao_list.append(       rt_step.step_dm_ao      )
        self._dm_orth_list.append(     rt_step.step_dm_orth    )
        self._fock_orth_list.append(   rt_step.step_fock_ao    )
        self._fock_ao_list.append(     rt_step.step_fock_orth  )

    def get_step_dict(self, iter_step=None, t=None):
        pass


    # def load_rt_step_index(self):
    #     return load_rt_step_index(self.chkfile)

    # def load_rt_step(self, step_index):
    #     if hasattr(step_index, '__iter__'):
    #         return [load_rt_step(self.chkfile, istep_index) for istep_index in step_index]
    #     else:
    #         return load_rt_step(self.chkfile, step_index)