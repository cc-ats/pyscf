# Author: Junjie Yang <yangjunjie0320@gmail.com> Zheng Pei and Yihan Shao
import unittest
import copy
import numpy
import scipy.linalg

from pyscf import gto, lib
from pyscf import scf, dft
from pyscf import rt

from pyscf.rt.rhf        import kernel
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

mol1.basis = '6-31g'
mol1.build()

rhf_631g = scf.RHF(mol1).run()
uhf_631g = scf.UHF(mol1).run()

class KnownValues(unittest.TestCase):
    def test_euler(self):
        for mf in [rhf_631g, uhf_631g]:
            mf   = rhf_631g
            mf.kernel()

            dm0  = mf.make_rdm1()
            rttd = rt.TDDFT(mf)
            prop_euler       = EulerPropagator(rttd, verbose=3)
            step_obj       = RealTimeStep(rttd,    verbose=3)
            step_obj.calculate_dipole = True
            step_obj.calculate_pop    = True
            step_obj.calculate_energy = True
            result_obj     = RealTimeResult(rttd,  verbose=3)
            result_obj._save_in_memory = True
            result_obj._save_in_disk   = False
            kernel(rttd, step_size = 0.02, total_step = 50, save_frequency = 1, dm_ao_init=dm0, result_obj=result_obj, prop_obj=prop_euler, step_obj = step_obj, verbose=4)

            e_ref           = mf.e_tot
            t_euler         = read_keyword_value("t",           result_obj=result_obj)
            e_elec_euler    = read_keyword_value("energy_elec", result_obj=result_obj)
            dm_orth_euler   = read_keyword_value("dm_orth",     result_obj=result_obj)
            fock_orth_euler = read_keyword_value("fock_orth",   result_obj=result_obj)
            dipole_euler    = read_keyword_value("dipole",      result_obj=result_obj)

            energy_euler    = read_keyword_value("energy_tot", result_obj=result_obj, chk_file=None)
            e_diff_list     = energy_euler-e_ref
            for e in e_diff_list:
                self.assertAlmostEqual(e, 0.0)

    def test_mmut(self):
        for mf in [rhf_631g, uhf_631g]:
            mf   = rhf_631g
            mf.kernel()

            dm0  = mf.make_rdm1()
            rttd = rt.TDDFT(mf)
            prop_mmut        = MMUTPropagator(rttd, verbose=3)
            step_obj       = RealTimeStep(rttd,    verbose=3)
            step_obj.calculate_dipole = True
            step_obj.calculate_pop    = True
            step_obj.calculate_energy = True
            result_obj     = RealTimeResult(rttd,  verbose=3)
            result_obj._save_in_memory = True
            result_obj._save_in_disk   = False
            kernel(rttd, step_size = 0.02, total_step = 50, save_frequency = 1, dm_ao_init=dm0, result_obj=result_obj, prop_obj=prop_mmut, step_obj = step_obj, verbose=4)

            e_ref          = mf.e_tot
            t_mmut         = read_keyword_value("t",           result_obj=result_obj)
            e_elec_mmut    = read_keyword_value("energy_elec", result_obj=result_obj)
            dm_orth_mmut   = read_keyword_value("dm_orth",     result_obj=result_obj)
            fock_orth_mmut = read_keyword_value("fock_orth",   result_obj=result_obj)
            dipole_mmut    = read_keyword_value("dipole",      result_obj=result_obj)

            energy_mmut     = read_keyword_value("energy_tot", result_obj=result_obj, chk_file=None)
            e_diff_list     = energy_mmut-e_ref
            for e in e_diff_list:
                self.assertAlmostEqual(e, 0.0)

    def test_eppc(self):
        for mf in [rhf_631g, uhf_631g]:
            mf   = rhf_631g
            dm0  = mf.make_rdm1()
            rttd = rt.TDDFT(mf)
            prop_eppc      = EPPCPropagator(rttd, verbose=3)
            step_obj       = RealTimeStep(rttd,    verbose=3)
            step_obj.calculate_dipole = True
            step_obj.calculate_pop    = True
            step_obj.calculate_energy = True
            result_obj     = RealTimeResult(rttd,  verbose=3)
            result_obj._save_in_memory = True
            result_obj._save_in_disk   = False
            kernel(rttd, step_size = 0.02, total_step = 50, save_frequency = 1, dm_ao_init=dm0, result_obj=result_obj, prop_obj=prop_eppc, step_obj = step_obj, verbose=4)

            e_ref          = mf.e_tot
            t_eppc         = read_keyword_value("t",           result_obj=result_obj)
            e_elec_eppc    = read_keyword_value("energy_elec", result_obj=result_obj)
            dm_orth_eppc   = read_keyword_value("dm_orth",     result_obj=result_obj)
            fock_orth_eppc = read_keyword_value("fock_orth",   result_obj=result_obj)
            dipole_eppc    = read_keyword_value("dipole",      result_obj=result_obj)

            energy_eppc     = read_keyword_value("energy_tot", result_obj=result_obj, chk_file=None)
            e_diff_list     = energy_eppc-e_ref
            for e in e_diff_list:
                self.assertAlmostEqual(e, 0.0)

    def test_lflp(self):
        for mf in [rhf_631g, uhf_631g]:
            mf   = rhf_631g
            dm0  = mf.make_rdm1()
            rttd = rt.TDDFT(mf)
            prop_lflp      = LFLPPCPropagator(rttd, verbose=3)
            step_obj       = RealTimeStep(rttd,    verbose=3)
            step_obj.calculate_dipole = True
            step_obj.calculate_pop    = True
            step_obj.calculate_energy = True
            result_obj     = RealTimeResult(rttd,  verbose=3)
            result_obj._save_in_memory = True
            result_obj._save_in_disk   = False
            kernel(rttd, step_size = 0.02, total_step = 50, save_frequency = 1, dm_ao_init=dm0, result_obj=result_obj, prop_obj=prop_lflp, step_obj = step_obj, verbose=4)

            e_ref          = mf.e_tot
            t_lflp         = read_keyword_value("t",           result_obj=result_obj)
            e_elec_lflp    = read_keyword_value("energy_elec", result_obj=result_obj)
            dm_orth_lflp   = read_keyword_value("dm_orth",     result_obj=result_obj)
            fock_orth_lflp = read_keyword_value("fock_orth",   result_obj=result_obj)
            dipole_lflp    = read_keyword_value("dipole",      result_obj=result_obj)

            energy_lflp     = read_keyword_value("energy_tot", result_obj=result_obj, chk_file=None)
            e_diff_list     = energy_lflp-e_ref
            for e in e_diff_list:
                self.assertAlmostEqual(e, 0.0)

if __name__ == "__main__":
    print("Full Tests for Propagators")
    unittest.main()