from pyscf.rt  import rhf as rhf_tdscf
from pyscf     import lib
from pyscf.lib import logger

class TDDFT(rhf_tdscf.TDHF):
    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** %s ********', self.__class__)
        log.info(
        'This is a real time TDSCF calculation initialized with a %s SCF',
            (
            "converged" if self.mf.converged else "not converged"
            )
        )
        if self.mf.converged:
            log.info(
                'The SCF converged tolerence is conv_tol = %g, conv_tol should be less that 1e-8'%self.mf.conv_tol
                )
        log.info(
            'The initial condition is a RKS instance'
            )
        log.info(
            'The xc functional is %s'%self.mf.xc
            )
        if self.chkfile:
            log.info('chkfile to save RT TDSCF result = %s', self.chkfile)
        log.info( 'dt = %f, maxstep = %d', self.dt, self.maxstep )
        log.info( 'prop_method = %s', self.prop_func.__name__)
        log.info('max_memory %d MB (current use %d MB)', self.max_memory, lib.current_memory()[0])

if __name__ == "__main__":
    from pyscf import dft, gto
    import numpy
    
    mol =   gto.Mole( atom='''
  H    0.0000000    0.0000000    0.3540000
  H    0.0000000    0.0000000   -0.3540000
    '''
    , basis='sto-3g', symmetry=False).build()

    mf = dft.RKS(mol)
    mf.verbose = 5
    mf.xc = "b3lyp"
    mf.kernel()

    dm = mf.make_rdm1()
    fock = mf.get_fock()

    rttd = TDDFT(mf)
    rttd.verbose = 5
    rttd.maxstep = 5
    rttd.prop_method = "lflp_pc"
    rttd.dt      = 0.2
    rttd.kernel(dm_ao_init=dm)
    print(rttd.netot)