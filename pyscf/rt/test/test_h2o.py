import unittest
import copy
import numpy
import scipy.linalg

from pyscf import gto, lib
from pyscf import scf, dft
from pyscf import rt

from pyscf.rt.result import read_keyword_value

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
        rttd.total_step     = 50
        rttd.step_size      = 0.02
        rttd.kernel(dm0)

        e_ref           = mf.e_tot
        energy_euler    = read_keyword_value("energy_tot", result_obj=rttd.result_obj, chk_file=None)
        e_diff_list     = energy_euler-e_ref
        for e in e_diff_list:
            self.assertAlmostEqual(e, 0.0)

    def test_rt_tdhf2(self):
        mf = rhf_dz
        dm0 = mf.make_rdm1()
        rttd = rt.TDDFT(mf)
        rttd.verbose        = 4
        rttd.total_step     = 50
        rttd.step_size      = 0.02
        rttd.kernel(dm0)

        e_ref           = mf.e_tot
        energy_euler    = read_keyword_value("energy_tot", result_obj=rttd.result_obj, chk_file=None)
        e_diff_list     = energy_euler-e_ref
        for e in e_diff_list:
            self.assertAlmostEqual(e, 0.0)

    def test_rt_tdhf3(self):
        mf = uhf_631g
        dm0 = mf.make_rdm1()
        rttd = rt.TDDFT(mf)
        rttd.verbose        = 4
        rttd.total_step     = 50
        rttd.step_size      = 0.02
        rttd.kernel(dm0)

        e_ref           = mf.e_tot
        energy_euler    = read_keyword_value("energy_tot", result_obj=rttd.result_obj, chk_file=None)
        e_diff_list     = energy_euler-e_ref
        for e in e_diff_list:
            self.assertAlmostEqual(e, 0.0)

    def test_rt_tdhf4(self):
        mf = uhf_dz
        dm0 = mf.make_rdm1()
        rttd = rt.TDDFT(mf)
        rttd.verbose        = 4
        rttd.total_step     = 50
        rttd.step_size      = 0.02
        rttd.kernel(dm0)

        e_ref           = mf.e_tot
        energy_euler    = read_keyword_value("energy_tot", result_obj=rttd.result_obj, chk_file=None)
        e_diff_list     = energy_euler-e_ref
        for e in e_diff_list:
            self.assertAlmostEqual(e, 0.0)

    def test_rt_tdks1(self):
        mf = rks_631g
        dm0 = mf.make_rdm1()
        rttd = rt.TDDFT(mf)
        rttd.verbose        = 4
        rttd.total_step     = 50
        rttd.step_size      = 0.02
        rttd.kernel(dm0)

        e_ref           = mf.e_tot
        energy_euler    = read_keyword_value("energy_tot", result_obj=rttd.result_obj, chk_file=None)
        e_diff_list     = energy_euler-e_ref
        for e in e_diff_list:
            self.assertAlmostEqual(e, 0.0)

    def test_rt_tdks2(self):
        mf = rks_dz
        dm0 = mf.make_rdm1()
        rttd = rt.TDDFT(mf)
        rttd.verbose        = 4
        rttd.total_step     = 50
        rttd.step_size      = 0.02
        rttd.kernel(dm0)

        e_ref           = mf.e_tot
        energy_euler    = read_keyword_value("energy_tot", result_obj=rttd.result_obj, chk_file=None)
        e_diff_list     = energy_euler-e_ref
        for e in e_diff_list:
            self.assertAlmostEqual(e, 0.0)

    def test_rt_tdks3(self):
        mf = uks_631g
        dm0 = mf.make_rdm1()
        rttd = rt.TDDFT(mf)
        rttd.verbose        = 4
        rttd.total_step     = 50
        rttd.step_size      = 0.02
        rttd.kernel(dm0)

        e_ref           = mf.e_tot
        energy_euler    = read_keyword_value("energy_tot", result_obj=rttd.result_obj, chk_file=None)
        e_diff_list     = energy_euler-e_ref
        for e in e_diff_list:
            self.assertAlmostEqual(e, 0.0)

    def test_rt_tdks4(self):
        mf = uks_dz
        dm0 = mf.make_rdm1()
        rttd = rt.TDDFT(mf)
        rttd.verbose        = 4
        rttd.total_step     = 50
        rttd.step_size      = 0.02
        rttd.kernel(dm0)

        e_ref           = mf.e_tot
        energy_euler    = read_keyword_value("energy_tot", result_obj=rttd.result_obj, chk_file=None)
        e_diff_list     = energy_euler-e_ref
        for e in e_diff_list:
            self.assertAlmostEqual(e, 0.0)

if __name__ == "__main__":
    print("Full Tests for H2O")
    unittest.main()
