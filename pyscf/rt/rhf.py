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

from pyscf.rt.propagator import euler_prop, mmut_prop
from pyscf.rt.propagator import ep_pc_prop, lflp_pc_prop

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

def kernel(tdscf, dm_ao_init= None, chk_file = None, calculate_dipole = None, calculate_pop =None):
    cput0 = (time.clock(), time.time())

    if dm_ao_init is None:  dm_ao_init = tdscf.dm_ao_init

    dt          = tdscf.dt
    maxstep     = tdscf.maxstep
    prop_func   = tdscf.prop_func
    orth_xtuple = tdscf._orth_xtuple 

    dm_ao_init     = dm_ao_init.astype(numpy.complex128)
    dm_prim_init   = tdscf.ao2orth_dm(dm_ao_init, orth_xtuple=orth_xtuple)

    h1e_ao_init    = tdscf.get_hcore(t=0.0)
    vhf_ao_init    = tdscf._scf.get_veff(dm_ao=dm_ao_init)

    fock_ao_init   = tdscf._scf.get_fock(h1e_ao_init, dm_ao=dm_ao_init, vhf=vhf_ao_init)
    fock_orth_init = tdscf.ao2orth_fock(fock_ao_init)

    etot_init      = tdscf._scf.energy_tot(dm=dm_ao_init, h1e=h1e_ao_init, vhf=vhf_ao_init).real

    shape = list(dm_ao_init.shape)
    
    ndm_prim[0]    = dm_prim_init
    nfock_prim[0]  = fock_prim_init
    ndm_ao[0]      = dm_ao_init
    nfock_ao[0]    = fock_ao_init
    netot[0]       = etot_init

    _temp_ts         = dt*numpy.array([0.0, 0.5, 1.0, 1.5, 2.0])
    logger.info(self, 'before building matrices, max_memory %d MB (current use %d MB)', self.max_memory, lib.current_memory()[0])
    temp_dm_orth   = numpy.zeros([5] + list(dm_ao_init.shape), dtype=numpy.complex128) # output
    temp_dm_ao     = numpy.zeros([5] + list(dm_ao_init.shape), dtype=numpy.complex128) # output
    temp_fock_orth = numpy.zeros([5] + list(dm_ao_init.shape), dtype=numpy.complex128) # output
    temp_fock_ao   = numpy.zeros([5] + list(dm_ao_init.shape), dtype=numpy.complex128) # output
    logger.info(self, 'after building matrices, max_memory %d MB (current use %d MB)', self.max_memory, lib.current_memory()[0])

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

        self._scf_get_veff     = None
        self._scf_get_fock     = None

# input parameters for propagation
# initial condtion
        self.dm_ao_init = None

# time step and maxstep
        self.dt         = None
        self.maxstep    = None
#
        self.calculate_dipole = True
        self.calculate_pop    = True

# propagation method
        self.prop_method     = None # a string
        self.prop_func       = None # a string
# electric field during propagation, a time-dependent electric field instance
        self.electric_field  = field
        self._get_field_ao   = None


    def prop_step(self, dt, fock_orth, dm_orth):
        return expia_b_exp_iat(-dt*fock_orth, dm_orth)

    def set_prop_func(self, key='euler'):
        '''
        In virtually all cases PC methods are superior in terms of stability.
        Others are perhaps only useful for debugging or simplicity.
        '''
        if (key is not None):
            if   (key.lower() == 'euler'):
                self.prop_func = euler_prop
            elif (key.lower() == 'mmut'):
                self.prop_func = mmut_prop
            elif (key.lower() == 'ep_pc_prop'):
                self.prop_func = ep_pc_prop
            elif (key.lower() == 'lflp_pc'):
                self.prop_func = lflp_pc_prop
            else:
                raise RuntimeError("unknown prop method!")
        else:
            self.prop_func = euler_prop

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

    def get_veff_ao(self, dm_orth=None, dm_orth_last=None, dm_ao=None, dm_ao_last=None,
                    vhf_last=None, orth_xtuple=None):
        assert (dm_orth is not None) and (dm_ao is not None)
        if orth_xtuple is None:
            orth_xtuple = self._orth_xtuple
        if (dm_ao is None) and (dm_orth is not None):
            dm_ao = self.orth2ao_dm(dm_orth, orth_xtuple=orth_xtuple)
        if (dm_orth_last is not None) and (dm_ao_last is None):
            dm_ao_last = self.orth2ao_dm(dm_orth, orth_xtuple=orth_xtuple)

        if dm_ao_last is not None:
            self._scf.direct_scf = True
            veff_ao = self._scf.get_veff(mol=self.mol, dm=dm_ao, dm_last=dm_ao_last, vhf_last=vhf_last, hermi=1)
            self._scf.direct_scf = False
            return veff_ao
        else:
            veff_ao = self._scf.get_veff(mol=self.mol, dm=dm_ao, hermi=1)
            return veff_ao

    def get_fock_ao(self, hcore_ao, dm_orth=None, dm_ao=None, veff_ao=None, orth_xtuple=None):
        assert (dm_orth is not None) and (dm_ao is not None)
        if orth_xtuple is None:
            orth_xtuple = self._orth_xtuple
        if (dm_ao is None) and (dm_orth is not None):
            dm_ao = self.orth2ao_dm(dm_orth, orth_xtuple=orth_xtuple)
        if veff_ao is None:
            veff_ao = self.get_veff_ao(dm_orth=dm_orth, dm_ao=dm_ao)
        return self._scf.get_fock(hcore_ao, self._ovlp_ao, veff_ao, dm_ao)

    def get_fock_orth(self, hcore_ao, dm_orth=None, dm_ao=None, veff_ao=None, orth_xtuple=None):
        assert (dm_orth is not None) and (dm_ao is not None)
        if orth_xtuple is None:
            orth_xtuple = self._orth_xtuple
        if (dm_ao is None) and (dm_orth is not None):
            dm_ao = self.orth2ao_dm(dm_orth, orth_xtuple=orth_xtuple)
        if veff_ao is None:
            veff_ao = self.get_veff_ao(dm_orth, dm_ao=dm_ao)
        return self.ao2orth_fock(
            self._scf.get_fock(hcore_ao, self._ovlp_ao, veff_ao, dm_ao),
            orth_xtuple=self._orth_xtuple
            )

    def get_energy_elec(self, hcore_ao, dm_orth,
                        dm_ao=None, veff_ao=None, orth_xtuple=None):
        if orth_xtuple is None:
            orth_xtuple = self._orth_xtuple
        if (dm_ao is None) and (dm_orth is not None):
            dm_ao = self.orth2ao_dm(dm_orth, orth_xtuple=orth_xtuple)
        if veff_ao is None:
            veff_ao = self.get_veff_ao(dm_orth, dm_ao=dm_ao, orth_xtuple=orth_xtuple)
        return self._scf.energy_elec(dm=dm_ao, h1e=hcore_ao, vhf=veff_ao).real

    def get_energy_tot(self, hcore_ao, dm_orth, dm_ao=None, veff_ao=None):
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

        if self.chkfile:
            log.info('chkfile to save RT TDSCF result = %s', self.chkfile)
        log.info( 'dt = %f, maxstep = %d', self.dt, self.maxstep )
        # log.info( 'prop_method = %s', self.prop_func.__name__)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])

    def _initialize(self):
        # mf information
        # self._scf.direct_scf   = True
        self._ovlp_ao          = self._scf.get_ovlp().astype(numpy.complex128)
        self._hcore_ao         = self._scf.get_hcore().astype(numpy.complex128)
        self._orth_xtuple = orth_canonical_mo(self._scf)

        if self.electric_field is not None:
            self._get_field_ao = self.electric_field.get_field_ao

        self.dump_flags()
        dump_rt_obj(self.chkfile, self)
        
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

    def kernel(self, dm_ao_init=None, chkfile=None, save_step=None):
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
           self, dm_ao_init, maxstep, 
            )
        logger.info(self, 'after propogation matrices, max_memory %d MB (current use %d MB)', self.max_memory, lib.current_memory()[0])
        logger.info(self, "Propagation finished")
        self._finalize()

    def dump_rt_step(self, idx, t, etot, dm_ao, dm_orth, fock_ao, fock_orth, dip=None, pop=None, field=None):
        rt_dic = {
            "t":t, "etot": etot, "dm_ao": dm_ao, "dm_orth": dm_orth, "fock_ao": fock_ao, "fock_orth": fock_orth
        }
        if dip is not None:
            rt_dic["dip"] = dip
        if pop is not None:
            rt_dic["pop"] = pop
        if field is not None:
            rt_dic["field"] = field
        dump_rt_step(self.chkfile, idx, **rt_dic)

    def load_rt_step_index(self):
        return load_rt_step_index(self.chkfile)

    def load_rt_step(self, step_index):
        if hasattr(step_index, '__iter__'):
            return [load_rt_step(self.chkfile, istep_index) for istep_index in step_index]
        else:
            return load_rt_step(self.chkfile, step_index)

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
    
    gau_vec = lambda t: gaussian_field_vec(t, 1.0, 1.0, 0.0, [0.020,0.00,0.00])
    gaussian_field = ClassicalElectricField(h2o, field_func=gau_vec, stop_time=10.0)

    rttd = TDHF(h2o_rhf, field=gaussian_field)
    rttd.verbose = 4
    rttd.maxstep = 10
    rttd.dt      = 0.02
    rttd._initialize()
    
    h1e = rttd.get_hcore_ao(5.0)
    veff_ao_0   = rttd.get_veff_ao(dm_orth_0, dm_ao=dm_0)
    veff_ao_1   = rttd.get_veff_ao(dm_orth_0, dm_ao=dm_0, dm_ao_last=dm_0)
    fock_orth_0 = rttd.get_fock_orth(h1e, dm_orth_0, dm_ao=dm_0, veff_ao=veff_ao_0)

    _chkfile    = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    tmp_chkfile = _chkfile.name

    dump_rt_obj(tmp_chkfile, rttd)

    for it, t in enumerate(numpy.linspace(0,100,1001)):
        rttd.dump_rt_step(
            it, t, 10.00, dm_0, dm_orth_0, fock_0, fock_orth_0,
            dip=[0,0,0], pop=None, field=[0,0,0]
            )

    step_index = rttd.load_rt_step_index()

    for step in step_index:
        print("step_index = ", step)
        rtstep = rttd.load_rt_step(step)
        print("t = %f"%rtstep["t"])
        print("field = ", rtstep["field"])
    
    rtstep = rttd.load_rt_step(range(100))
    print("step_index = ", rtstep[10])
    print("t = %f"%rtstep[10]["t"])
