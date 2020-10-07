import unittest
import copy
import numpy
import scipy.linalg

from pyscf import gto, lib
from pyscf import scf, dft
from pyscf import rt

from pyscf.rt.rhf        import kernel
from pyscf.rt.util       import print_cx_matrix
from pyscf.rt.propagator import EulerPropagator, MMUTPropagator
from pyscf.rt.propagator import EPPCPropagator, LFLPPCPropagator
from pyscf.rt.result     import RealTimeStep, RealTimeResult
from pyscf.rt.result     import read_index_list, read_step_dict, read_keyword_value

mol1 = gto.Mole()
mol1.verbose = 7
mol1.output = '/dev/null'
mol1.atom = [
    ["O" , (0. , 0.     , 0.)],
    [1   , (0. , -0.757 , 0.587)],
    [1   , (0. , 0.757  , 0.587)] ]

mol1.basis = {"H": '6-31g',
             "O": '6-31g',}
mol1.build()

rhf_631g = scf.RHF(mol1).run()
rks_631g = scf.RKS(mol1).run()
uhf_631g = scf.UHF(mol1).run()
uks_631g = scf.UKS(mol1).run()

mol2 = gto.Mole()
mol2.verbose = 7
mol2.output = '/dev/null'
mol2.atom = [
    ["O" , (0. , 0.     , 0.)],
    [1   , (0. , -0.757 , 0.587)],
    [1   , (0. , 0.757  , 0.587)] ]

mol2.basis = "cc-pVDZ"
mol2.build()

rhf_dz = scf.RHF(mol2).run()
rks_dz = scf.RKS(mol2).run()
uhf_dz = scf.UHF(mol2).run()
uks_dz = scf.UKS(mol2).run()


class KnownValues(unittest.TestCase):
    def test_rt_tdhf1(self):
        mf = rhf_631g
        dm0 = mf.make_rdm1()
        rttd = rt.TDDFT(mf)
        rttd.verbose        = 4
        rttd.total_step     = None
        rttd.step_size      = None

        prop_euler       = EulerPropagator(rttd, verbose=3)
        step_obj_1       = RealTimeStep(rttd,    verbose=3)
        step_obj_1.calculate_dipole = True
        step_obj_1.calculate_pop    = True
        step_obj_1.calculate_energy = True
        result_obj_1     = RealTimeResult(rttd,  verbose=3)
        result_obj_1._save_in_memory = True
        result_obj_1._save_in_disk   = False
        kernel(rttd, step_size = 0.02, total_step = 50, save_frequency = 1, dm_ao_init=dm0,
                    result_obj=result_obj_1, prop_obj=prop_euler, step_obj = step_obj_1, verbose=4)

        e_ref           = mf.e_tot
        energy_euler    = read_keyword_value("energy_tot", result_obj=result_obj_1)
        e_diff_list     = energy_euler-e_ref
        for e in e_diff_list:
            self.assertAlmostEqual(e, 0.0)

    def test_rt_tdhf2(self):
        mf = rks_631g
        dm0 = mf.make_rdm1()
        rttd = rt.TDDFT(mf)
        rttd.verbose        = 4
        rttd.total_step     = None
        rttd.step_size      = None

        prop_euler       = EulerPropagator(rttd, verbose=3)
        step_obj_1       = RealTimeStep(rttd,    verbose=3)
        step_obj_1.calculate_dipole = True
        step_obj_1.calculate_pop    = True
        step_obj_1.calculate_energy = True
        result_obj_1     = RealTimeResult(rttd,  verbose=3)
        result_obj_1._save_in_memory = True
        result_obj_1._save_in_disk   = False
        kernel(rttd, step_size = 0.02, total_step = 50, save_frequency = 1, dm_ao_init=dm0,
                    result_obj=result_obj_1, prop_obj=prop_euler, step_obj = step_obj_1, verbose=4)

        e_ref           = mf.e_tot
        energy_euler    = read_keyword_value("energy_tot", result_obj=result_obj_1)
        e_diff_list     = energy_euler-e_ref
        for e in e_diff_list:
            self.assertAlmostEqual(e, 0.0)

if __name__ == "__main__":
    print("Full Tests for H2O vdz")
    unittest.main()
