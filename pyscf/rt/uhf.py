# Author: Junjie Yang <yangjunjie0320@gmail.com> Zheng Pei and Yihan Shao

import time
import tempfile

from functools import reduce

import numpy
from numpy import asarray, complex128, dot

import scipy

from pyscf import gto, scf
from pyscf import lib, lo
from pyscf.lib import logger
from pyscf.rt import chkfile

from pyscf.rt import rhf as rhf_tdscf
from pyscf.rt.propagator import Propogator, PCPropogator
from pyscf.rt.propagator import EulerPropogator, MMUTPropogator
from pyscf.rt.propagator import EPPCPropogator, LFLPPCPropogator

from pyscf.rt.result import RealTimeStep, RealTimeResult

from pyscf.rt.util import print_matrix, print_cx_matrix
from pyscf.rt.util import expia_b_exp_ia

from pyscf import __config__

REF_BASIS         = getattr(__config__, 'lo_orth_pre_orth_ao_method', 'ANO'      )
MUTE_CHKFILE      = getattr(__config__, 'rt_tdscf_mute_chkfile',      False      )
PRINT_MAT_NCOL    = getattr(__config__, 'rt_tdscf_print_mat_ncol',    7          )
ORTH_METHOD       = getattr(__config__, 'rt_tdscf_orth_ao_method',    'canonical')

# re-define Orthogonalize AOs for UHF
def orth_canonical_mo(scf_obj, ovlp_ao=None):
    """ transform AOs """
    assert isinstance(scf_obj, scf.uhf.UHF)
    logger.info(scf_obj, "the AOs are orthogonalized with unrestricted canonical MO coefficients")
    if not scf_obj.converged:
        logger.warn(scf_obj,"the SCF object must be converged")
    if ovlp_ao is None:
        ovlp_ao = scf_obj.get_ovlp()
        ovlp_ao = asarray(ovlp_ao, dtype=complex128)

    x       = asarray(scf_obj.mo_coeff, dtype=complex128)
    x_t     = x.transpose(0,2,1)
    x_inv   = dot(x_t, ovlp_ao)
    x_t_inv = x_inv.transpose(0,2,1)
    orth_xtuple = (x, x_t, x_inv, x_t_inv)
    return orth_xtuple
    
def ao2orth_contravariant(contravariant_matrix_ao, orth_xtuple):
    """ transform contravariant matrix from orthogonal basis to AO basis """
    x_inv   = orth_xtuple[2]
    x_t_inv = orth_xtuple[3]
    contravariant_matrix_orth_a = reduce(dot, [x_inv[0], contravariant_matrix_ao[0], x_t_inv[0]])
    contravariant_matrix_orth_b = reduce(dot, [x_inv[1], contravariant_matrix_ao[0], x_t_inv[1]])
    return asarray([contravariant_matrix_orth_a, contravariant_matrix_orth_b])

def orth2ao_contravariant(contravariant_matrix_orth, orth_xtuple):
    """ transform contravariant matrix from AO to orthogonal basis """
    x   = orth_xtuple[0]
    x_t = orth_xtuple[1]
    contravariant_matrix_ao_a = reduce(dot, [x[0], contravariant_matrix_orth[0], x_t[0]])
    contravariant_matrix_ao_b = reduce(dot, [x[1], contravariant_matrix_orth[1], x_t[1]])
    return asarray([contravariant_matrix_ao_a, contravariant_matrix_ao_b])

def ao2orth_covariant(covariant_matrix_ao, orth_xtuple):
    """ transform covariant matrix from AO to orthogonal basis """
    x   = orth_xtuple[0]
    x_t = orth_xtuple[1]
    covariant_matrix_orth_a = reduce(dot, [x_t[0], covariant_matrix_ao[0], x[0]])
    covariant_matrix_orth_b = reduce(dot, [x_t[1], covariant_matrix_ao[1], x[1]])
    return asarray([covariant_matrix_orth_a, covariant_matrix_orth_b])

def orth2ao_covariant(covariant_matrix_orth, orth_xtuple):
    """ transform covariant matrix from orthogonal basis to AO basis """
    x_inv   = orth_xtuple[2]
    x_t_inv = orth_xtuple[3]
    covariant_matrix_ao_a = reduce(dot, [x_t_inv[0], covariant_matrix_orth[0], x_inv[0]])
    covariant_matrix_ao_b = reduce(dot, [x_t_inv[1], covariant_matrix_orth[1], x_inv[1]])
    return asarray([covariant_matrix_ao_a, covariant_matrix_ao_b])

# propagate step
def prop_step(tdscf, dt, fock_prim, dm_prim):
    propogator_a = expm(-1j*dt*fock_prim[0])
    propogator_b = expm(-1j*dt*fock_prim[1])

    dm_prim_a_   = reduce(numpy.dot, [propogator_a, dm_prim[0], propogator_a.conj().T])
    # dm_prim_a_   = (dm_prim_a_ + dm_prim_a_.conj().T)/2
    dm_prim_b_   = reduce(numpy.dot, [propogator_b, dm_prim[1], propogator_b.conj().T])
    # dm_prim_b_   = (dm_prim_b_ + dm_prim_b_.conj().T)/2

    dm_prim_     = numpy.array((dm_prim_a_, dm_prim_b_))
    dm_ao_       = tdscf.orth2ao_dm(  dm_prim_)
    
    return dm_prim_, dm_ao_
   
class TDHF(rhf_tdscf.TDHF):
    def propagate_step(self, step_size, fock_orth, dm_orth, orth_xtuple=None):
        if orth_xtuple is None:
            orth_xtuple = self._orth_xtuple
        dm_orth_   = asarray(
            [expia_b_exp_ia(-step_size*fock_orth[0], dm_orth[0]), expia_b_exp_ia(-step_size*fock_orth[1], dm_orth[1])]
            )
        dm_ao_     = self.orth2ao_dm(dm_orth_, orth_xtuple=orth_xtuple)
        return dm_orth_, dm_ao_

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

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** %s ********', self.__class__)
        log.info(
        'This is a Real-Time TDSCF calculation initialized with a %s UHF instance',
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

        if self.prop_obj is None:   self.set_prop_obj(key=self.prop_method)
        if self.step_obj is None:   self.step_obj   = RealTimeStep(self)
        if self.result_obj is None: self.result_obj   = RealTimeResult(self)

        self.dump_flags()

if __name__ == "__main__":
    from pyscf import gto, scf
    from field import ClassicalElectricField, constant_field_vec, gaussian_field_vec
    h2o =   gto.Mole( atom='''
    O     0.00000000    -0.00001441    -0.34824012
    H    -0.00000000     0.76001092    -0.93285191
    H     0.00000000    -0.75999650    -0.93290797
    '''
    , basis='sto-3g', symmetry=False).build()

    h2o_uhf    = scf.UHF(h2o)
    h2o_uhf.verbose = 4
    h2o_uhf.conv_tol = 1e-20
    h2o_uhf.max_cycle = 200
    h2o_uhf.kernel()

    dm_0   = h2o_uhf.make_rdm1()
    fock_0 = h2o_uhf.get_fock()

    orth_xtuple = orth_canonical_mo(h2o_uhf)
    dm_orth_0   = ao2orth_contravariant(dm_0, orth_xtuple)
    fock_orth_0 = ao2orth_covariant(fock_0, orth_xtuple)

    print_cx_matrix("dm_orth_0_alpha   = ", dm_orth_0[0])
    print_cx_matrix("fock_orth_0_alpha = ", fock_orth_0[0])

    print_cx_matrix("dm_orth_0_beta    = ", dm_orth_0[1])
    print_cx_matrix("fock_orth_0_beta  = ", fock_orth_0[1])

    gaussian_vec = lambda t: gaussian_field_vec(t, 0.5329, 1.0, 0.0, [1e-2, 0.0, 0.0])
    gaussian_field = ClassicalElectricField(h2o, field_func=gaussian_vec, stop_time=10.0)

    rttd = TDHF(h2o_uhf, field=gaussian_field)
    rttd.verbose        = 3
    rttd.total_step     = 10
    rttd.step_size      = 0.02
    rttd.prop_method    = "lflp-pc"
    rttd.save_frequency = 5
    rttd.kernel(dm_ao_init=dm_0)

    for i in range(3):
        print("")
        print("#####################################")
        print("t = %f"%rttd.result_obj._time_list[i])
        print("field = ", gaussian_vec(rttd.result_obj._time_list[i]))
        print_cx_matrix("dm_orth_alpha   = ", rttd.result_obj._dm_orth_list[i][0])
        # print_cx_matrix("fock_orth_alpha = ", rttd.result_obj._fock_orth_list[i][0])
        print_cx_matrix("dm_orth_beta    = ", rttd.result_obj._dm_orth_list[i][1])
        # print_cx_matrix("fock_orth_beta  = ", rttd.result_obj._fock_orth_list[i][1])
