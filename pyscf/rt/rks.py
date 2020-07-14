# Author: Junjie Yang <yangjunjie0320@gmail.com> Zheng Pei and Yihan Shao

from pyscf.rt  import rhf as rhf_tdscf
from pyscf     import lib
from pyscf.lib import logger

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
        log.info( 'xc = %s', self._scf.xc )
        log.info( 'step_size = %f, total_step = %d', self.step_size, self.total_step )
        log.info( 'prop_method = %s', self.prop_method)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])

if __name__ == "__main__":
    ''' This is a short test.'''
    from pyscf import gto, scf
    from result import read_step_dict
    from field import ClassicalElectricField, constant_field_vec, gaussian_field_vec
    import numpy

    h2o =   gto.Mole( atom='''
    O     0.00000000    -0.00001441    -0.34824012
    H    -0.00000000     0.76001092    -0.93285191
    H     0.00000000    -0.75999650    -0.93290797
    '''
    , basis='sto-3g', symmetry=False).build() # water

    h2o_rks    = scf.RKS(h2o)
    h2o_rks.xc = "B3LYP"
    h2o_rks.verbose = 0
    h2o_rks.conv_tol = 1e-12
    h2o_rks.kernel()
    dm_init = h2o_rks.make_rdm1()

    gaussian_vec = lambda t: gaussian_field_vec(t, 0.5329, 1.0, 0.0, [1e-2, 0.0, 0.0])
    gaussian_field = ClassicalElectricField(h2o, field_func=gaussian_vec, stop_time=10.0)

    rttd = TDDFT(h2o_rks, field=gaussian_field)
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