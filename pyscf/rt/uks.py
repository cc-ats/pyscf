# Author: Junjie Yang <yangjunjie0320@gmail.com> Zheng Pei and Yihan Shao

from pyscf.rt  import uhf as uhf_tdscf
from pyscf     import lib
from pyscf.lib import logger

class TDDFT(uhf_tdscf.TDHF):
    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** %s ********', self.__class__)
        log.info(
        'This is a real time TDSCF calculation initialized with a %s SCF',
            (
            "converged" if self._scf.converged else "not converged"
            )
        )
        if self._scf.converged:
            log.info(
                'The SCF converged tolerence is conv_tol = %g, conv_tol should be less that 1e-8'%self._scf.conv_tol
                )
        log.info(
            'The initial condition is a UKS instance'
            )
        log.info(
            'The xc functional is %s'%self._scf.xc
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
    O    0.0000000    0.0000000    0.5754646
    O    0.0000000    0.0000000   -0.5754646
    '''
    , basis='sto-3g', spin=2, symmetry=False).build()

    mf = dft.UKS(mol)
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