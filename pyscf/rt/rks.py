# Author: Junjie Yang <yangjunjie0320@gmail.com> Zheng Pei and Yihan Shao

from pyscf.rt  import rhf as rhf_tdscf
from pyscf     import lib
from pyscf.lib import logger

from pyscf.rt.util import print_matrix, print_cx_matrix

class TDDFT(rhf_tdscf.TDHF):
    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** %s ********', self.__class__)
        log.info(
        'This is a Real-Time TDDFT calculation initialized with a %s RKS instance',
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

if __name__ == "__main__":
    from pyscf import gto, scf
    from field import ClassicalElectricField, constant_field_vec, gaussian_field_vec
    h2o =   gto.Mole( atom='''
    O     0.00000000    -0.00001441    -0.34824012
    H    -0.00000000     0.76001092    -0.93285191
    H     0.00000000    -0.75999650    -0.93290797
    '''
    , basis='sto-3g', symmetry=False).build() # water

    h2o_rks    = scf.RKS(h2o)
    h2o_rks.xc = "B3LYP"
    h2o_rks.verbose = 4
    h2o_rks.conv_tol = 1e-12
    h2o_rks.kernel()

    dm_0   = h2o_rks.make_rdm1()
    fock_0 = h2o_rks.get_fock()

    gaussian_vec   = lambda t: gaussian_field_vec(t, 0.5329, 1.0, 0.0, [1e-2, 0.0, 0.0])
    gaussian_field = ClassicalElectricField(h2o, field_func=gaussian_vec, stop_time=10.0)

    rttd = TDDFT(h2o_rks, field=gaussian_field)
    rttd.verbose        = 3
    rttd.total_step     = 10
    rttd.step_size      = 0.02
    rttd.prop_method    = "mmut"
    rttd.kernel(dm_ao_init=dm_0)

    for i in range(10):
        print("")
        print("#####################################")
        print("t = %f"%rttd.result_obj._time_list[i])
        print("field = ", gaussian_vec(rttd.result_obj._time_list[i]))
        print_cx_matrix("dm_orth = ", rttd.result_obj._dm_orth_list[i])
        print_cx_matrix("fock_orth = ", rttd.result_obj._fock_orth_list[i])