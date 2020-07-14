# Author: Junjie Yang <yangjunjie0320@gmail.com> Zheng Pei and Yihan Shao
import time
import tempfile
import numpy
import scipy

from functools import reduce
from numpy import dot, complex128

from pyscf import lib
from pyscf.lib import logger

from pyscf.rt.propagator import Propogator,      PCPropogator
from pyscf.rt.propagator import EulerPropogator, MMUTPropogator
from pyscf.rt.propagator import EPPCPropogator,  LFLPPCPropogator

from pyscf.rt.result import RealTimeStep,    RealTimeResult
from pyscf.rt.result import read_index_list, read_step_dict         

from pyscf.rt.util import print_matrix, print_cx_matrix
from pyscf.rt.util import expia_b_exp_ia

from pyscf import __config__

def orth_canonical_mo(scf_obj, ovlp_ao=None):
    """ get transformation matrices """
    logger.info(scf_obj, "the AOs are orthogonalized with restricted canonical MO coefficients")
    if not scf_obj.converged:
        logger.warn(scf_obj,"the SCF object must be converged")
    if ovlp_ao is None:
        ovlp_ao = scf_obj.get_ovlp()

    c_orth = scf_obj.mo_coeff
    x       = c_orth.astype(complex128)
    x_t     = x.T
    x_inv   = dot(x_t, ovlp_ao)
    x_t_inv = x_inv.T
    orth_xtuple = (x, x_t, x_inv, x_t_inv)
    return orth_xtuple

def ao2orth_contravariant(contravariant_matrix_ao, orth_xtuple):
    """ transform contravariant matrix from orthogonal basis to AO basis """
    x_inv   = orth_xtuple[2]
    x_t_inv = orth_xtuple[3]
    contravariant_matrix_orth = reduce(dot, [x_inv, contravariant_matrix_ao, x_t_inv])
    return contravariant_matrix_orth

def orth2ao_contravariant(contravariant_matrix_orth, orth_xtuple):
    """ transform contravariant matrix from AO to orthogonal basis """
    x   = orth_xtuple[0]
    x_t = orth_xtuple[1]
    contravariant_matrix_ao = reduce(dot, [x, contravariant_matrix_orth, x_t])
    return contravariant_matrix_ao

def ao2orth_covariant(covariant_matrix_ao, orth_xtuple):
    """ transform covariant matrix from AO to orthogonal basis """
    x   = orth_xtuple[0]
    x_t = orth_xtuple[1]
    covariant_matrix_orth = reduce(dot, [x_t, covariant_matrix_ao, x])
    return covariant_matrix_orth

def orth2ao_covariant(covariant_matrix_orth, orth_xtuple):
    """ transform covariant matrix from orthogonal basis to AO basis """
    x_inv   = orth_xtuple[2]
    x_t_inv = orth_xtuple[3]
    covariant_matrix_ao = reduce(dot, [x_t_inv, covariant_matrix_orth, x_inv])
    return covariant_matrix_ao

def kernel(rt_obj, dm_ao_init= None, dm_orth_init=None, step_size = None, total_step = None,
                   prop_obj = None, step_obj = None, result_obj = None, save_frequency = None,
                   verbose = None):

    cput0 = (time.clock(), time.time())

    if dm_ao_init is None:
        if dm_orth_init is not None:
            dm_ao_init = rt_obj.orth2ao_dm(dm_orth_init)
        else:
            dm_ao_init   = rt_obj.dm_ao_init
            dm_orth_init = rt_obj.ao2orth_dm(dm_ao_init)
    else:
        dm_orth_init = rt_obj.ao2orth_dm(dm_ao_init)

    if total_step is None:
        total_step     = rt_obj.total_step

    if step_size is None:
        step_size      = rt_obj.step_size
    
    if prop_obj    is None:  prop_obj    = rt_obj.prop_obj
    if step_obj    is None:  step_obj    = rt_obj.step_obj
    if result_obj  is None:  result_obj     = rt_obj.result_obj

    h1e_ao_init    = rt_obj.get_hcore_ao(0.0)
    vhf_ao_init    = rt_obj.get_veff_ao(dm_orth=dm_orth_init ,dm_ao=dm_ao_init)

    fock_ao_init   = rt_obj.get_fock_ao(hcore_ao=h1e_ao_init, dm_orth=dm_orth_init ,dm_ao=dm_ao_init)
    fock_orth_init = rt_obj.get_fock_orth(hcore_ao=h1e_ao_init, fock_ao=fock_ao_init, 
                                          dm_orth=dm_orth_init ,dm_ao=dm_ao_init)

    step_obj._initialize(dm_ao_init, dm_orth_init, fock_ao_init, fock_orth_init,
                       h1e_ao_init, vhf_ao_init)

    step_iter = prop_obj.first_step(dm_ao_init, dm_orth_init, fock_ao_init, fock_orth_init,
                   step_obj=step_obj, verbose=verbose)
    save_iter = result_obj._initialize(step_obj)
    cput1 = logger.timer(rt_obj, 'Initialize rt_obj', *cput0)

    while step_iter < total_step:
        # propagation step
        step_iter = prop_obj.propagate_step(step_obj=step_obj, verbose=verbose)
        if step_iter%save_frequency == 0:
            result_obj._update(step_obj)
    result_obj._finalize()
    cput2 = logger.timer(rt_obj, 'Finish rt_obj', *cput0)


class TDHF(lib.StreamObject):
    def __init__(self, scf_obj, field=None):
        """ the class that defines the system, mol and scf_method """
        self._scf           = scf_obj
        self.mol            = scf_obj.mol
        self.verbose        = scf_obj.verbose
        self._scf.verbose   = 0
        self.mol.verbose    = 0

        if not scf_obj.converged:
            logger.warn(self,
            "SCF not converged, RT-TDSCF method should be initialized with a converged scf instance."
            )

        self.max_memory     = scf_obj.max_memory
        self.stdout         = scf_obj.stdout

# the chkfile will be removed automatically, to save the chkfile, assign a
# filename to self.chk_file
        self._chkfile   = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        self.chk_file   = self._chkfile.name
        self.step_obj   = None

# propagation method
        self.prop_method     = None # a string
        self.prop_obj        = None # a Propogator object

# intermediate result 
        self._orth_xtuple     = None
        self._ovlp_ao         = None
        self._hcore_ao        = None

# input parameters for propagation
# initial condition
        self.dm_ao_init = None
# time step and total_step
        self.step_size     = None
        self.total_step    = None
#
        self.calculate_dipole  = True
        self.calculate_pop     = True
        self.calculate_energy  = True
        self.save_frequency   = None
        self.save_in_memory   = True
        self.save_in_disk     = False

# electric field during propagation, a time-dependent electric field object
        self.electric_field  = field

# result
        self.result_obj        = None
        self.save_index_list   = None

    def propagate_step(self, step_size, fock_orth, dm_orth, orth_xtuple=None):
        if orth_xtuple is None:
            orth_xtuple = self._orth_xtuple
        dm_orth_   = expia_b_exp_ia(-step_size*fock_orth, dm_orth)
        dm_ao_     = self.orth2ao_dm(dm_orth_, orth_xtuple=orth_xtuple)
        return dm_orth_, dm_ao_

    def set_prop_obj(self, key='euler'):
        '''
        In virtually all cases PC methods are superior in terms of stability.
        Others are perhaps only useful for debugging or simplicity.
        '''
        if (key is not None):
            if   (key.lower() == 'euler'):
                # self.prop_func = euler_prop
                self.prop_obj = EulerPropogator(self, verbose=self.verbose)
            elif (key.lower() == 'mmut'):
                self.prop_obj = MMUTPropogator(self, verbose=self.verbose)
            elif (key.lower() == 'ep-pc' or key.lower() == 'eppc'):
                self.prop_obj = EPPCPropogator(self, verbose=self.verbose)
            elif (key.lower() == 'lflp-pc' or key.lower() == 'lflp'):
                self.prop_obj = LFLPPCPropogator(self, verbose=self.verbose)
            else:
                raise RuntimeError("unknown prop method!")
        else:
            self.prop_obj = EulerPropogator(self)

    def ao2orth_dm(self, dm_ao, orth_xtuple=None):
        '''
        Transfrom density matrix from AO basis to orthogonal basis
        '''
        if orth_xtuple is None:
            orth_xtuple = self._orth_xtuple
        return ao2orth_contravariant(dm_ao, orth_xtuple)

    def orth2ao_dm(self, dm_orth, orth_xtuple=None):
        if orth_xtuple is None:
            orth_xtuple = self._orth_xtuple
        return orth2ao_contravariant(dm_orth, orth_xtuple)

    def ao2orth_fock(self, fock_ao, orth_xtuple=None):
        if orth_xtuple is None:
            orth_xtuple = self._orth_xtuple
        return ao2orth_covariant(fock_ao, orth_xtuple)

    def orth2ao_fock(self, fock_orth, orth_xtuple=None):
        if orth_xtuple is None:
            orth_xtuple = self._orth_xtuple
        return orth2ao_covariant(fock_orth, orth_xtuple)

    def get_hcore_ao(self, t):
        if self.electric_field is None:
            return self._hcore_ao
        else:
            return self._hcore_ao + self.electric_field.get_field_ao(t)

    def get_veff_ao(self, dm_orth=None, dm_ao=None, orth_xtuple=None):
        assert (dm_orth is not None) or (dm_ao is not None)
        if orth_xtuple is None:
            orth_xtuple = self._orth_xtuple
        if (dm_ao is None) and (dm_orth is not None):
            dm_ao = self.orth2ao_dm(dm_orth, orth_xtuple=orth_xtuple)
        veff_ao = self._scf.get_veff(mol=self.mol, dm=dm_ao, hermi=1)
        return veff_ao

    def get_fock_ao(self, hcore_ao, dm_orth=None, dm_ao=None, veff_ao=None, orth_xtuple=None):
        assert (dm_orth is not None) or (dm_ao is not None)
        if orth_xtuple is None:
            orth_xtuple = self._orth_xtuple
        if (dm_ao is None) and (dm_orth is not None):
            dm_ao = self.orth2ao_dm(dm_orth, orth_xtuple=orth_xtuple)
        if veff_ao is None:
            veff_ao = self.get_veff_ao(dm_orth=dm_orth, dm_ao=dm_ao)
        fock_ao = self._scf.get_fock(hcore_ao, self._ovlp_ao, veff_ao, dm_ao)
        return fock_ao

    def get_fock_orth(self, hcore_ao, fock_ao=None, dm_orth=None, dm_ao=None, veff_ao=None, orth_xtuple=None):
        if fock_ao is None:
            assert (dm_orth is not None) or (dm_ao is not None)
            if orth_xtuple is None:
                orth_xtuple = self._orth_xtuple
            if (dm_ao is None) and (dm_orth is not None):
                dm_ao = self.orth2ao_dm(dm_orth, orth_xtuple=orth_xtuple)
            if veff_ao is None:
                veff_ao = self.get_veff_ao(dm_orth, dm_ao=dm_ao)
            fock_ao =  self.ao2orth_fock(
            self._scf.get_fock(hcore_ao, self._ovlp_ao, veff_ao, dm_ao),
            orth_xtuple=orth_xtuple
            )
        fock_orth = self.ao2orth_fock(fock_ao, orth_xtuple=orth_xtuple)
        return fock_orth

    def get_energy_elec(self, hcore_ao, dm_orth=None,
                        dm_ao=None, veff_ao=None, orth_xtuple=None):
        if orth_xtuple is None:
            orth_xtuple = self._orth_xtuple
        if (dm_ao is None) and (dm_orth is not None):
            dm_ao = self.orth2ao_dm(dm_orth, orth_xtuple=orth_xtuple)
        if veff_ao is None:
            veff_ao = self.get_veff_ao(dm_orth, dm_ao=dm_ao, orth_xtuple=orth_xtuple)
        return self._scf.energy_elec(dm=dm_ao, h1e=hcore_ao, vhf=veff_ao)[0].real

    def get_energy_tot(self, hcore_ao, dm_orth=None, dm_ao=None, veff_ao=None, orth_xtuple=None):
        if orth_xtuple is None:
            orth_xtuple = self._orth_xtuple
        if (dm_ao is None) and (dm_orth is not None):
            dm_ao = self.orth2ao_dm(dm_orth, orth_xtuple=orth_xtuple)
        if veff_ao is None:
            veff_ao = self.get_veff_ao(dm_orth=dm_orth, dm_ao=dm_ao)
        return self._scf.energy_tot(dm=dm_ao, h1e=hcore_ao, vhf=veff_ao).real

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** %s ********', self.__class__)
        log.info(
        'This is a Real-Time TDSCF calculation initialized with a %s RHF instance',
            ("converged" if self._scf.converged else "not converged")
        )
        if self._scf.converged:
            log.info(
                'The SCF converged tolerence is conv_tol = %g'%self._scf.conv_tol
                )

        if self.chk_file:
            log.info('chkfile to save RT TDSCF result = %s', self.chk_file)
        log.info('step_size = %f, total_step = %d', self.step_size, self.total_step)
        log.info('prop_method = %s', self.prop_method)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])

    def _initialize(self):
        self._ovlp_ao          = self._scf.get_ovlp().astype(complex128)
        self._hcore_ao         = self._scf.get_hcore().astype(complex128)
        self._orth_xtuple      = orth_canonical_mo(self._scf)
        self.dump_flags()

    def _finalize(self):
        self.save_index_list = read_index_list(result_obj=self.result_obj , chk_file=self.chk_file)

    def kernel(self, dm_ao_init= None, dm_orth_init=None, step_size = None, total_step = None,
                   save_frequency = None, prop_method = None,
                   save_in_disk = None, save_in_memory = None, chk_file = None,
                   calculate_dipole = None, calculate_pop =None, calculate_energy=None,
                   verbose = None):
        self._initialize()

        if verbose is None:
            verbose = self.verbose

        if dm_ao_init is None:
            if dm_orth_init is not None:
                dm_ao_init = self.orth2ao_dm(dm_orth_init)
            else:
                dm_ao_init   = self.dm_ao_init
                dm_orth_init = self.ao2orth_dm(dm_ao_init)
        else:
            dm_orth_init = self.ao2orth_dm(dm_ao_init)

        if prop_method is None:
            prop_method = self.prop_method
        self.set_prop_obj(key=prop_method)
        prop_obj    = self.prop_obj

        if calculate_dipole is None:  calculate_dipole    = self.calculate_dipole
        if calculate_pop    is None:  calculate_pop       = self.calculate_pop
        if calculate_energy is None:  calculate_energy    = self.calculate_energy

        if self.step_obj is None:
            self.step_obj   = RealTimeStep(self, verbose=verbose)
            self.step_obj.calculate_dipole = calculate_dipole
            self.step_obj.calculate_pop    = calculate_pop
            self.step_obj.calculate_energy = calculate_energy
        step_obj    = self.step_obj

        if save_in_memory is None:  save_in_memory = self.save_in_memory
        if chk_file       is None:  chk_file     = self.chk_file
        if save_in_disk   is None:  save_in_disk = self.save_in_disk

        if self.result_obj is None:
            self.result_obj   = RealTimeResult(self, verbose=verbose)
            self.result_obj._save_in_disk   = save_in_disk
            self.result_obj._chk_file       = chk_file
            self.result_obj._save_in_memory = save_in_memory
        result_obj    = self.result_obj

        if total_step is None:
            total_step     = self.total_step

        if save_frequency is None and self.save_frequency is not None:
            save_frequency = self.save_frequency
        else:
            save_frequency = 1

        kernel(
           self, dm_ao_init= dm_ao_init, dm_orth_init=dm_orth_init,
                 step_size = step_size, total_step = total_step, save_frequency = save_frequency,
                 prop_obj = prop_obj, step_obj = step_obj, result_obj = result_obj,
                 verbose = verbose
            )
        logger.info(self, 'after propogation matrices, max_memory %d MB (current use %d MB)',
                    self.max_memory, lib.current_memory()[0])
        logger.info(self, "Propagation finished")
        self._finalize()

    def read_step_dict(self, index, chk_file=None):
        if chk_file is None:
            chk_file = self.chk_file
        return read_step_dict(index, result_obj=self.result_obj, chk_file=chk_file)

if __name__ == "__main__":
    ''' This is a short test.'''
    from pyscf import gto, scf
    from result import read_step_dict
    from field import ClassicalElectricField, constant_field_vec, gaussian_field_vec

    h2o =   gto.Mole( atom='''
    O     0.00000000    -0.00001441    -0.34824012
    H    -0.00000000     0.76001092    -0.93285191
    H     0.00000000    -0.75999650    -0.93290797
    '''
    , basis='sto-3g', symmetry=False).build() # water

    h2o_rhf    = scf.RHF(h2o)
    h2o_rhf.verbose = 0
    h2o_rhf.conv_tol = 1e-12
    h2o_rhf.kernel()
    dm_init = h2o_rhf.make_rdm1()

    gaussian_vec = lambda t: gaussian_field_vec(t, 0.5329, 1.0, 0.0, [1e-2, 0.0, 0.0])
    gaussian_field = ClassicalElectricField(h2o, field_func=gaussian_vec, stop_time=10.0)

    rttd = TDHF(h2o_rhf, field=gaussian_field)
    rttd.verbose        = 5
    rttd.total_step     = 10
    rttd.step_size      = 0.02
    rttd.chk_file       = "./test/h2o_rt.chk"
    rttd.prop_method    = "eppc"
    rttd.save_frequency = 5
    rttd.kernel(dm_ao_init=dm_init, save_in_disk = True, save_in_memory = True,
                calculate_energy=True, calculate_dipole=True)

    for i in rttd.save_index_list:
        print("t = %f"%rttd.result_obj._time_list[i])
        temp_dict = read_step_dict(i, result_obj = None, chk_file = "./test/h2o_rt.chk" )
        assert numpy.allclose(temp_dict["dm_orth"], rttd.result_obj._dm_orth_list[i])
        temp_dict = read_step_dict(i, result_obj = rttd.result_obj, chk_file = None )
        assert numpy.allclose(temp_dict["dm_orth"], rttd.result_obj._dm_orth_list[i])
        temp_dict = rttd.read_step_dict(i)
        assert numpy.allclose(temp_dict["dm_orth"], rttd.result_obj._dm_orth_list[i])
