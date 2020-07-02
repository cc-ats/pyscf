# Author: Junjie Yang <yangjunjie0320@gmail.com> Zheng Pei and Yihan Shao
import time
import tempfile
import numpy
import scipy

from functools import reduce
from numpy import dot

from pyscf import gto, scf
from pyscf.scf.hf import get_jk

from pyscf import lib, lo
from pyscf.lib import logger
from pyscf.rt.chkfile import dump_rt_obj, dump_rt_step, load_rt_step, load_rt_step_index

from pyscf.rt.propagator import EulerPropogator, MMUTPropogator
from pyscf.rt.propagator import EPPCPropogator, LFLPPCPropogator

from pyscf.rt.result import RealTimeStep, RealTimeResult

from pyscf.rt.util import print_matrix, print_cx_matrix
from pyscf.rt.util import errm, expm, expia_b_exp_ia

from pyscf import __config__

PRINT_MAT_NCOL    = getattr(__config__, 'rt_tdscf_print_mat_ncol',              7)
PC_TOL            = getattr(__config__, 'rt_tdscf_pc_tol',                  1e-14)
PC_MAX_ITER       = getattr(__config__, 'rt_tdscf_pc_max_iter',                20)

LAST      = 0 
LAST_HALF = 1
THIS      = 2
NEXT_HALF = 3
NEXT      = 4

def orth_canonical_mo(scf_obj, ovlp_ao=None):
    """ transform AOs """
    logger.info(scf_obj, "the AOs are orthogonalized with canonical MO coefficients")
    if not scf_obj.converged:
        logger.warn(scf_obj,"the SCF object must be converged")
    if ovlp_ao is None:
        ovlp_ao = scf_obj.get_ovlp()

    c_orth = scf_obj.mo_coeff
    x       = c_orth.astype(numpy.complex128)
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

def kernel(rt_obj, dm_ao_init= None, dm_orth_init=None,
                   step_size = None, total_step = None,
                   prop_obj = None, step_obj = None, save_frequency = None,
                   result_obj = None, save_in_memory = None,
                   chk_file = None,  save_in_disk = None,
                   calculate_dipole = None, calculate_pop =None, calculate_energy=None,
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

    if prop_obj   is None:  prop_obj   = rt_obj.prop_obj
    if step_obj    is None:  step_obj    = rt_obj.step_obj

    if calculate_dipole is None:  calculate_dipole = rt_obj.calculate_dipole
    if calculate_pop    is None:  calculate_pop   = rt_obj.calculate_pop
    if calculate_energy is None:  calculate_energy    = rt_obj.calculate_energy

    if total_step is None:
        total_step     = rt_obj.total_step
    if save_frequency is not None:
        total_save_step = total_step//save_frequency
    else:
        total_save_step = total_step

    if save_frequency is None and rt_obj.save_frequency is not None:
        save_frequency = rt_obj.save_frequency
    else:
        save_frequency = 1
    if result_obj     is None:  result_obj  = rt_obj.result_obj
    if save_in_memory is None:  save_in_memory = rt_obj.save_in_memory

    if save_in_memory:
        assert result_obj is not None
        logger.info(rt_obj, "The results would be saved in the memory, %s", result_obj)

    if chk_file       is None:  chk_file  = rt_obj.chk_file
    if save_in_disk   is None:  save_in_disk = rt_obj.save_in_disk

    if save_in_disk:
        assert chk_file is not None
        logger.info(rt_obj, "The results would be saved in the disk, %s", chk_file)

    assert save_in_disk or save_in_memory

    h1e_ao_init    = rt_obj.get_hcore_ao(t=0.0)
    vhf_ao_init    = rt_obj.get_veff_ao(dm_orth=dm_orth_init ,dm_ao=dm_ao_init)

    fock_ao_init   = rt_obj.get_fock_ao(hcore_ao=h1e_ao_init, dm_orth=dm_orth_init ,dm_ao=dm_ao_init)
    fock_orth_init = rt_obj.get_fock_orth(hcore_ao=h1e_ao_init, fock_ao=fock_ao_init, 
                                          dm_orth=dm_orth_init ,dm_ao=dm_ao_init)
# TODO: initialize the step_obj here
    step_obj._initialize(dm_ao_init, dm_orth_init, fock_ao_init, fock_orth_init,
                       h1e_ao_init, vhf_ao_init,
                       calculate_dipole=calculate_dipole, calculate_pop=calculate_pop,
                       calculate_energy=calculate_energy
                       )
# TODO: initialize the propagator here
    step_iter = prop_obj.first_step(dm_ao_init, dm_orth_init, fock_ao_init, fock_orth_init,
                   step_obj=step_obj, verbose=verbose)
# TODO: initialize the result_obj here
    save_iter = result_obj._initialize(step_obj)

    cput1 = logger.timer(rt_obj, 'initialize rt_obj', *cput0)

    while step_iter <= total_step:
        # propagation step
        step_iter = prop_obj.propagate_step(step_obj=step_obj, verbose=verbose)
        if step_iter%save_frequency == 0:
            result_obj._update(step_obj)
        

    # cput2 = logger.timer(tdscf, 'propagation %d time steps'%(istep-1), *cput0)

    # if (do_dump_chk) and (tdscf.chkfile) and (tdscf.save_frequency is None):
    #     ntime      = tdscf.ntime
    #     netot      = tdscf.netot
    #     ndm_ao     = tdscf.ndm_ao
    #     tdscf.dump_chk(locals())
    #     cput3 = logger.timer(tdscf, 'dump chk finished', *cput0)
    # elif (do_dump_chk) and (tdscf.chkfile) and (tdscf.save_frequency is not None):
    #     logger.note(tdscf, 'The results are saved in every %d steps.', tdscf.save_frequency)
    #     ntime      = tdscf.ntime[::tdscf.save_frequency]
    #     netot      = tdscf.netot[::tdscf.save_frequency]
    #     ndm_ao     = tdscf.ndm_ao[::tdscf.save_frequency]
    #     # print(locals())
    #     tdscf.dump_chk(locals())
    #     cput3 = logger.timer(tdscf, 'dump chk finished', *cput0)

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

        self.save_frequency   = None
        self.save_in_memory   = True
        self.save_in_disk     = False

        self.result_obj        = None
        self.step_obj          = None

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

# propagation method
        self.prop_method     = None # a string
        self.prop_obj        = None # a Propogator object

# electric field during propagation, a time-dependent electric field object
        self.electric_field  = field
        self._get_field_ao   = None

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
                self.prop_obj = EulerPropogator(self)
            elif (key.lower() == 'mmut'):
                self.prop_obj = MMUTPropogator(self)
            elif (key.lower() == 'ep-pc'):
                self.prop_obj = EPPCPropogator(self)
            elif (key.lower() == 'lflp-pc'):
                self.prop_obj = LFLPPCPropogator(self)
            else:
                raise RuntimeError("unknown prop method!")
        else:
            self.prop_obj = EulerPropogator(self)

    def ao2orth_dm(self, dm_ao, orth_xtuple=None):
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

    def get_hcore_ao(self, t, get_field_ao=None):
        if get_field_ao is None:
            if self._get_field_ao is None:
                return self._hcore_ao
            else:
                self._hcore_ao + self._get_field_ao(t)
        else:
            return self._hcore_ao + get_field_ao(t)

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
        return self._scf.get_fock(hcore_ao, self._ovlp_ao, veff_ao, dm_ao)

    def get_fock_orth(self, hcore_ao, fock_ao=None, dm_orth=None, dm_ao=None, veff_ao=None, orth_xtuple=None):
        assert (dm_orth is not None) or (dm_ao is not None)
        if orth_xtuple is None:
            orth_xtuple = self._orth_xtuple
        if (dm_ao is None) and (dm_orth is not None):
            dm_ao = self.orth2ao_dm(dm_orth, orth_xtuple=orth_xtuple)
        if veff_ao is None:
            veff_ao = self.get_veff_ao(dm_orth, dm_ao=dm_ao)
        if fock_ao is None:
            fock_ao =  self.ao2orth_fock(
            self._scf.get_fock(hcore_ao, self._ovlp_ao, veff_ao, dm_ao),
            orth_xtuple=orth_xtuple
            )
        return fock_ao

    def get_energy_elec(self, hcore_ao, dm_orth=None,
                        dm_ao=None, veff_ao=None, orth_xtuple=None):
        if orth_xtuple is None:
            orth_xtuple = self._orth_xtuple
        if (dm_ao is None) and (dm_orth is not None):
            dm_ao = self.orth2ao_dm(dm_orth, orth_xtuple=orth_xtuple)
        if veff_ao is None:
            veff_ao = self.get_veff_ao(dm_orth, dm_ao=dm_ao, orth_xtuple=orth_xtuple)
        return self._scf.energy_elec(dm=dm_ao, h1e=hcore_ao, vhf=veff_ao)[0].real

    def get_energy_tot(self, hcore_ao, dm_orth=None, dm_ao=None, veff_ao=None):
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
        log.info( 'step_size = %f, total_step = %d', self.step_size, self.total_step )
        log.info( 'prop_obj = %s', self.prop_obj.__class__.__name__)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])

    def _initialize(self):
        self._ovlp_ao          = self._scf.get_ovlp().astype(numpy.complex128)
        self._hcore_ao         = self._scf.get_hcore().astype(numpy.complex128)
        self._orth_xtuple      = orth_canonical_mo(self._scf)

        if self.electric_field is not None:
            self._get_field_ao = self.electric_field.get_field_ao

        self.set_prop_obj(key=self.prop_method)
        self.step_obj   = RealTimeStep(self)
        self.result_obj = RealTimeResult(self)
        self.dump_flags()
        # dump_rt_obj(self.chk_file, self)
        
        if self.verbose >= logger.DEBUG:
            print_matrix(
                "XT S X", reduce(dot, (self._orth_xtuple[1], self._ovlp_ao, self._orth_xtuple[0]))
                , ncols=PRINT_MAT_NCOL)

    def _finalize(self):
        pass
    '''
        logger.info(self, "Finalization begins here")
        logger.info(self, "Finalization finished")
    '''

    def kernel(self, dm_ao_init= None, dm_orth_init= None,
                   step_size = None, total_step = None,
                   prop_obj = None, step_obj = None, save_frequency = None,
                   result_obj = None, save_in_memory = None,
                   chk_file = None,  save_in_disk = None,
                   calculate_dipole = None, calculate_pop =None, calculate_energy=None,
                   verbose = None):
        self._initialize()
        if dm_ao_init is None:
            if self.dm_ao_init is not None:
                dm_ao_init = self.dm_ao_init
            else:
                dm_ao_init = self._scf.make_rdm1()
        logger.info(self, "Propagation begins here")
        if self.verbose >= logger.DEBUG1:
            print_matrix("The initial density matrix is, ", dm_ao_init, ncols=PRINT_MAT_NCOL)

        kernel(
           self, dm_ao_init, total_step, 
            )
        logger.info(self, 'after propogation matrices, max_memory %d MB (current use %d MB)', self.max_memory, lib.current_memory()[0])
        logger.info(self, "Propagation finished")
        self._finalize()

    # def dump_rt_step(self, idx, t, etot, dm_ao, dm_orth, fock_ao, fock_orth,
    #                  dip=None, pop=None, field=None,
    #                  save_in_memory=None, save_in_disk=None):

    #     if save_in_memory is None:
    #         save_in_memory = self.save_in_memory
    #     if save_in_disk is None:
    #         save_in_disk = self.save_in_disk

    #     rt_dic = {
    #         "t":t, "etot": etot, "dm_ao": dm_ao, "dm_orth": dm_orth, "fock_ao": fock_ao, "fock_orth": fock_orth
    #     }
    #     if dip is not None:
    #         rt_dic["dip"] = dip
    #     if pop is not None:
    #         rt_dic["pop"] = pop
    #     if field is not None:
    #         rt_dic["field"] = field

    #     if (save_in_memory is True) and (save_in_disk is False):
    #         pass
    #     else:
    #         pass

    #     dump_rt_step(self.chk_file, idx, **rt_dic)

if __name__ == "__main__":
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

    dm_0   = h2o_rhf.make_rdm1()
    fock_0 = h2o_rhf.get_fock()

    orth_xtuple = orth_canonical_mo(h2o_rhf)
    dm_orth_0   = ao2orth_contravariant(dm_0, orth_xtuple)
    fock_orth_0 = ao2orth_covariant(fock_0, orth_xtuple)
    
    gau_vec = lambda t: [0.0, 0.0, 0.0]
    print("gau_vec(0) = ", gau_vec(0))
    gaussian_field = ClassicalElectricField(h2o, field_func=gau_vec, stop_time=10.0)

    rttd = TDHF(h2o_rhf, field=gaussian_field)
    rttd.verbose        = 4
    rttd.total_step     = 10
    rttd.step_size      = 0.002
    rttd._initialize()
    kernel(rttd, dm_ao_init=dm_0)

    for i, dm_orth in enumerate(rttd.result_obj._dm_orth_list):
        print("")
        print("#####################################")
        print("t = %f"%rttd.result_obj._time_list[i])
        print_cx_matrix("dm_orth = ", dm_orth)
        print_cx_matrix("fock_orth = ", rttd.result_obj._fock_orth_list[i])
