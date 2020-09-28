#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy
import scipy.linalg
import tempfile
from pyscf import gto
from pyscf import scf
from pyscf import dft

h2o_z0 = gto.M(
    verbose = 5,
    output = '/dev/null',
    atom = [
    ["O" , (0. , 0.     , 0.)],
    [1   , (0. , -0.757 , 0.587)],
    [1   , (0. , 0.757  , 0.587)] ],
    basis = '6-31g')

def tearDownModule():
    global h2o_z0
    h2o_z0.stdout.close()
    del h2o_z0

class KnownValues(unittest.TestCase):
    def test_nr_rhf(self):
        mf = scf.RHF(h2o_z0)
        mf.max_cycle = 1
        mf.conv_check = False
        mf.kernel()
        nr = scf.gdm(mf)
        nr.max_cycle = 2
        nr.conv_tol_grad = 1e-5
        self.assertAlmostEqual(nr.kernel(), -75.98394849812, 9)

    def test_nr_rks_lda(self):
        mf = dft.RKS(h2o_z0)
        eref = mf.kernel()
        mf.max_cycle = 1
        mf.conv_check = False
        mf.kernel()
        nr = scf.newton(mf)
        nr.max_cycle = 3
        nr.conv_tol_grad = 1e-5
        self.assertAlmostEqual(nr.kernel(), eref, 9)

    def test_nr_rks_rsh(self):
        '''test range-separated Coulomb'''
        mf = dft.RKS(h2o_z0)
        mf.xc = 'wb97x'
        eref = mf.kernel()
        mf.max_cycle = 1
        mf.conv_check = False
        mf.kernel()
        nr = scf.newton(mf)
        nr.max_cycle = 3
        nr.conv_tol_grad = 1e-5
        self.assertAlmostEqual(nr.kernel(), eref, 9)

    def test_nr_rks(self):
        mf = dft.RKS(h2o_z0)
        mf.xc = 'b3lyp'
        eref = mf.kernel()
        mf.max_cycle = 1
        mf.conv_check = False
        mf.kernel()
        nr = scf.newton(mf)
        nr.max_cycle = 3
        nr.conv_tol_grad = 1e-5
        self.assertAlmostEqual(nr.kernel(), eref, 9)

    def test_with_df(self):
        mf = scf.RHF(h2o_z0).density_fit().newton().run()
        self.assertTrue(mf._eri is None)
        self.assertTrue(mf._scf._eri is None)
        self.assertAlmostEqual(mf.e_tot, -75.983944727996, 9)

        mf = scf.RHF(h2o_z0).newton().density_fit().run()
        self.assertTrue(mf._eri is None)
        self.assertTrue(mf._scf._eri is not None)
        self.assertAlmostEqual(mf.e_tot, -75.9839484980661, 9)

if __name__ == "__main__":
    print("Full Tests for Newton solver")
    unittest.main()

