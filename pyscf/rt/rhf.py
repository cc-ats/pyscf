# Author: Junjie Yang <yangjunjie0320@gmail.com> Zheng Pei and Yihan Shao
import time
import tempfile
import numpy
import scipy

from functools import reduce
from numpy import dot

from pyscf import gto, scf
from pyscf import lib, lo
from pyscf.lib import logger
from pyscf.rt  import chkfile

# from pyscf.rt.propagator import euler_prop, mmut_prop
# from pyscf.rt.propagator import ep_pc_prop, lflp_pc_prop

from pyscf.rt.util import build_absorption_spectrum
from pyscf.rt.util import print_matrix, print_cx_matrix
from pyscf.rt.util import errm, expm, expia_b_exp_iat

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

def kernel(tdscf,              dm_ao_init= None,
           ndm_prim  = None, nfock_prim  = None, #output
           ndm_ao    = None, nfock_ao    = None, #output
           netot     = None, do_dump_chk = True
           ):
    cput0 = (time.clock(), time.time())

    if dm_ao_init is None:  dm_ao_init = tdscf.dm_ao_init

    dt        = tdscf.dt
    maxstep   = tdscf.maxstep
    prop_func = tdscf.prop_func

    if ndm_prim is None:
        ndm_prim = tdscf.ndm_prim
    if nfock_prim is None:
        nfock_prim = tdscf.nfock_prim
    if ndm_ao is None:
        ndm_ao = tdscf.ndm_ao
    if nfock_ao is None:
        nfock_ao = tdscf.nfock_ao

    dm_ao_init     = dm_ao_init.astype(numpy.complex128)
    dm_prim_init   = tdscf.ao2orth_dm(dm_ao_init)

    h1e_ao_init    = tdscf.get_hcore(t=0.0)
    vhf_ao_init    = tdscf._scf.get_veff(dm=dm_ao_init)

    fock_ao_init   = tdscf._scf.get_fock(dm=dm_ao_init, h1e=h1e_ao_init, vhf=vhf_ao_init)
    fock_prim_init = tdscf.ao2orth_fock(fock_ao_init)

    etot_init      = tdscf._scf.energy_tot(dm=dm_ao_init, h1e=h1e_ao_init, vhf=vhf_ao_init).real

    shape = list(dm_ao_init.shape)
    
    ndm_prim[0]    = dm_prim_init
    nfock_prim[0]  = fock_prim_init
    ndm_ao[0]      = dm_ao_init
    nfock_ao[0]    = fock_ao_init
    netot[0]       = etot_init

    _temp_ts         = dt*numpy.array([0.0, 0.5, 1.0, 1.5, 2.0])
    _temp_dm_prims   = numpy.zeros([5] + shape, dtype=numpy.complex128)
    _temp_fock_prims = numpy.zeros([5] + shape, dtype=numpy.complex128)
    _temp_dm_aos     = numpy.zeros([5] + shape, dtype=numpy.complex128)
    _temp_fock_aos   = numpy.zeros([5] + shape, dtype=numpy.complex128)

    cput1 = logger.timer(tdscf, 'initialize td-scf', *cput0)

# propagation start here
# TODO: the initializations should be more careful
    _temp_dm_prims[LAST]   = ndm_prim[0]
    _temp_fock_prims[LAST] = nfock_prim[0]
    _temp_dm_aos[LAST]     = ndm_ao[0]
    _temp_fock_aos[LAST]   = nfock_ao[0]

    _temp_dm_prims[LAST_HALF]   = ndm_prim[0]
    _temp_fock_prims[LAST_HALF] = nfock_prim[0]
    _temp_dm_aos[LAST_HALF]     = ndm_ao[0]
    _temp_fock_aos[LAST_HALF]   = nfock_ao[0]

    _temp_dm_prims[THIS]   = ndm_prim[0]
    _temp_fock_prims[THIS] = nfock_prim[0]
    _temp_dm_aos[THIS]     = ndm_ao[0]
    _temp_fock_aos[THIS]   = nfock_ao[0]

    istep = 1
    while istep <= maxstep:
        if istep%100 ==1:
            logger.note(tdscf, 'istep=%d, time=%f, delta e=%e',
            istep-1, tdscf.ntime[istep-1], tdscf.netot[istep-1]-tdscf.netot[0])
        # propagation step
        netot[istep] = tdscf.prop_func(tdscf,  _temp_ts,
               _temp_dm_prims,   _temp_dm_aos,
               _temp_fock_prims, _temp_fock_aos)
        ndm_prim[istep]   =   _temp_dm_prims[THIS]
        ndm_ao[istep]     =     _temp_dm_aos[THIS]
        nfock_prim[istep] = _temp_fock_prims[THIS]
        nfock_ao[istep]   =   _temp_fock_aos[THIS]
        _temp_ts = _temp_ts + dt
        istep += 1

    cput2 = logger.timer(tdscf, 'propagation %d time steps'%(istep-1), *cput0)

    if (do_dump_chk) and (tdscf.chkfile) and (tdscf.save_step is None):
        ntime      = tdscf.ntime
        netot      = tdscf.netot
        ndm_ao     = tdscf.ndm_ao
        tdscf.dump_chk(locals())
        cput3 = logger.timer(tdscf, 'dump chk finished', *cput0)
    elif (do_dump_chk) and (tdscf.chkfile) and (tdscf.save_step is not None):
        logger.note(tdscf, 'The results are saved in every %d steps.', tdscf.save_step)
        ntime      = tdscf.ntime[::tdscf.save_step]
        netot      = tdscf.netot[::tdscf.save_step]
        ndm_ao     = tdscf.ndm_ao[::tdscf.save_step]
        # print(locals())
        tdscf.dump_chk(locals())
        cput3 = logger.timer(tdscf, 'dump chk finished', *cput0)

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
# filename to self.chkfile
        self._chkfile   = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        self.chkfile    = self._chkfile.name
        self.save_step  = None

# intermediate result 
        self._orth_xtuple     = None
        self._ovlp_ao         = None
        self._hcore_ao        = None

# input parameters for propagation
# initial condtion
        self.dm_ao_init = None

# time step and maxstep
        self.dt         = None
        self.maxstep    = None

# propagation method
        self.prop_method     = None # a string
# electric field during propagation, a time-dependent electric field instance
        self.electric_field  = field

# don't modify the following attributes, they are not input options
        # self.nstep       = None
        # self.ntime       = None
        # self.ndm_prim    = None
        # self.nfock_prim  = None
        # self.ndm_ao      = None
        # self.nfock_ao    = None
        # self.netot       = None

    def prop_step(self, dt, fock_prim, dm_prim):
        return expia_b_exp_iat(-dt*fock_prim, dm_prim)

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

    def prop_func(self):
        pass

    def get_hcore_ao(self, t, electric_field=None):
        if electric_field is None:
            electric_field = self.electric_field

        if electric_field is None:
            return self._hcore_ao
        else:
            return self._hcore_ao + electric_field.get_field_ao(t)

    def get_veff_ao(self, dm_orth, dm_ao=None):
        if dm_ao is None:
            dm_ao = self.orth2ao_dm(dm_orth, orth_xtuple=self._orth_xtuple)
        return self._scf.get_veff(dm_ao)

    def get_fock_ao(self, hcore_ao, dm_orth, dm_ao=None, veff_ao=None):
        if dm_ao is None:
            dm_ao = self.orth2ao_dm(dm_orth, orth_xtuple=self._orth_xtuple)
        if veff_ao is None:
            veff_ao = self.get_veff_ao(dm_orth, dm_ao=dm_ao)
        return hcore_ao + veff_ao

    def get_fock_orth(self, hcore_ao, dm_orth, dm_ao=None, veff_ao=None):
        if dm_ao is None:
            dm_ao = self.orth2ao_dm(dm_orth, orth_xtuple=self._orth_xtuple)
        if veff_ao is None:
            veff_ao = self.get_veff_ao(dm_orth, dm_ao=dm_ao)
        return self.ao2orth_fock(hcore_ao + veff_ao, orth_xtuple=self._orth_xtuple)

    def get_energy_elec(self, hcore_ao, dm_orth, dm_ao=None, veff_ao=None):
        if dm_ao is None:
            dm_ao = self.orth2ao_dm(dm_orth, orth_xtuple=self._orth_xtuple)
        if veff_ao is None:
            veff_ao = self.get_veff_ao(dm_orth, dm_ao=dm_ao)
        return self._scf.energy_elec(dm=dm_ao, h1e=hcore_ao, vhf=veff_ao)

    def get_energy_tot(self, hcore_ao, dm_orth, dm_ao=None, veff_ao=None):
        if dm_ao is None:
            dm_ao = self.orth2ao_dm(dm_orth, orth_xtuple=self._orth_xtuple)
        if veff_ao is None:
            veff_ao = self.get_veff_ao(dm_orth, dm_ao=dm_ao)
        return self._scf.energy_tot(dm=dm_ao, h1e=hcore_ao, vhf=veff_ao)

        # if (self.efield_vec is None) or (t is None):
        #     return self.hcore_ao
        # else:
        #     if self.ele_dip_ao is None:
        #         # the interaction between the system and electric field
        #         self.ele_dip_ao      = self._scf.mol.intor_symmetric('int1e_r', comp=3)
            
        #     h = self.hcore_ao + numpy.einsum(
        #         'xij,x->ij', self.ele_dip_ao, self.efield_vec(t)
        #         ).astype(numpy.complex128)
        #     return h

    # def set_prop_func(self, key='euler'):
    #     '''
    #     In virtually all cases PC methods are superior in terms of stability.
    #     Others are perhaps only useful for debugging or simplicity.
    #     '''
    #     if (key is not None):
    #         if   (key.lower() == 'euler'):
    #             self.prop_func = euler_prop
    #         elif (key.lower() == 'mmut'):
    #             self.prop_func = mmut_prop
    #         elif (key.lower() == 'amut1'):
    #             self.prop_func = amut1_prop
    #         elif (key.lower() == 'amut2'):
    #             self.prop_func = amut2_prop
    #         elif (key.lower() == 'amut3'):
    #             self.prop_func = amut3_prop
    #         elif (key.lower() == 'amut_pc'):
    #             self.prop_func = amut_pc_prop
    #         elif (key.lower() == 'ep_pc'):
    #             self.prop_func = ep_pc_prop
    #         elif (key.lower() == 'lflp_pc'):
    #             self.prop_func = lflp_pc_prop
    #         else:
    #             raise RuntimeError("unknown prop method!")
    #     else:
    #         self.prop_func = euler_prop

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** %s ********', self.__class__)
        log.info(
        'This is a Real-Time TDSCF calculation initialized with a %s SCF',
            ("converged" if self._scf.converged else "not converged")
        )
        if self._scf.converged:
            log.info(
                'The SCF converged tolerence is conv_tol = %g, conv_tol should be less that 1e-8'%self._scf.conv_tol
                )
        log.info(
            'The initial condition is a RHF instance'
            )
        if self.chkfile:
            log.info('chkfile to save RT TDSCF result = %s', self.chkfile)
        log.info( 'dt = %f, maxstep = %d', self.dt, self.maxstep )
        log.info( 'prop_method = %s', self.prop_func.__name__)
        log.info('max_memory %d MB (current use %d MB)', self.max_memory, lib.current_memory()[0])

    def _initialize(self):
        # mf information
        self._ovlp_ao         = self._scf.get_ovlp().astype(numpy.complex128)
        self._hcore_ao        = self._scf.get_hcore().astype(numpy.complex128)
        self._orth_xtuple = orth_canonical_mo(self._scf)
        # self.dump_flags()
        
        if self.verbose >= logger.DEBUG1:
            print_matrix(
                "XT S X", reduce(dot, (self._orth_xtuple[1], self._ovlp_ao, self._orth_xtuple[0]))
                , ncols=PRINT_MAT_NCOL)

    def _finalize(self):
        self.ndipole = numpy.zeros([self.maxstep+1,             3])
        self.npop    = numpy.zeros([self.maxstep+1, self.mol.natm])
        logger.info(self, "Finalization begins here")
        s1e = self._scf.get_ovlp()
        for i,idm in enumerate(self.ndm_ao):
            self.ndipole[i] = self._scf.dip_moment(dm = idm.real, unit='au', verbose=0)
            self.npop[i]    = self._scf.mulliken_pop(dm = idm.real, s=s1e, verbose=0)[1]
        logger.info(self, "Finalization finished")

    def kernel(self, dm_ao_init=None, do_dump_chk=True):
        self._initialize()
        if dm_ao_init is None:
            if self.dm_ao_init is not None:
                dm_ao_init = self.dm_ao_init
            else:
                dm_ao_init = self._scf.make_rdm1()
        logger.info(self, "Propagation begins here")
        if self.verbose >= logger.DEBUG1:
            print_matrix("The initial density matrix is, ", dm_ao_init, ncols=PRINT_MAT_NCOL)

        logger.info(self, 'before building matrices, max_memory %d MB (current use %d MB)', self.max_memory, lib.current_memory()[0])
        self.nstep      = numpy.linspace(0, self.maxstep, self.maxstep+1, dtype=int) # output
        self.ntime      = self.dt*self.nstep                                      # output
        self.ndm_prim   = numpy.zeros([self.maxstep+1] + list(dm_ao_init.shape), dtype=numpy.complex128) # output
        self.ndm_ao     = numpy.zeros([self.maxstep+1] + list(dm_ao_init.shape), dtype=numpy.complex128) # output
        self.nfock_prim = numpy.zeros([self.maxstep+1] + list(dm_ao_init.shape), dtype=numpy.complex128) # output
        self.nfock_ao   = numpy.zeros([self.maxstep+1] + list(dm_ao_init.shape), dtype=numpy.complex128) # output
        self.netot      = numpy.zeros([self.maxstep+1])                                # output
        logger.info(self, 'after building matrices, max_memory %d MB (current use %d MB)', self.max_memory, lib.current_memory()[0])
        kernel(
           self,                      dm_ao_init  = dm_ao_init,
           ndm_prim  = self.ndm_prim, nfock_prim  = self.nfock_prim, #output
           ndm_ao    = self.ndm_ao,   nfock_ao    = self.nfock_ao,   #output
           netot     = self.netot,    do_dump_chk = do_dump_chk
            )
        logger.info(self, 'after propogation matrices, max_memory %d MB (current use %d MB)', self.max_memory, lib.current_memory()[0])
        logger.info(self, "Propagation finished")
        self._finalize()

#TODO: !!!!
    def dump_step(self, envs):
        # print('ndm_ao shape is ', envs['ndm_ao'].shape)
        if self.chkfile:
            logger.info(self, 'chkfile to save RT TDSCF result is %s', self.chkfile)
            chkfile.dump_rt(
                self.mol, self.chkfile,
                envs['ntime'], envs['netot'], envs['ndm_ao'],
                overwrite_mol=False)

    def dump_chk(self, envs):
        # print('ndm_ao shape is ', envs['ndm_ao'].shape)
        if self.chkfile:
            logger.info(self, 'chkfile to save RT TDSCF result is %s', self.chkfile)
            chkfile.dump_rt(
                self.mol, self.chkfile,
                envs['ntime'], envs['netot'], envs['ndm_ao'],
                overwrite_mol=False)

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
