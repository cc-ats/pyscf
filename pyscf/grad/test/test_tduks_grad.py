#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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
from pyscf import lib
from pyscf import gto, dft
from pyscf import tdscf
from pyscf.grad import tduks as tduks_grad

mol = gto.Mole()
mol.verbose = 5
mol.output = '/dev/null'
mol.atom = [
    ['H' , (0. , 0. , 1.804)],
    ['F' , (0. , 0. , 0.)], ]
mol.unit = 'B'
mol.charge = 2
mol.spin = 2
mol.basis = '631g'
mol.build()
pmol = mol.copy()
mf_lda = dft.UKS(mol).set(xc='LDA,', conv_tol=1e-12)
mf_lda.kernel()
mf_gga = dft.UKS(mol).set(xc='b88,', conv_tol=1e-12)
mf_gga.kernel()

def tearDownModule():
    global mol, pmol
    mol.stdout.close()
    del mol, pmol

class KnownValues(unittest.TestCase):
    def test_tda_lda(self):
        td = tdscf.TDA(mf_lda).run(nstates=3)
        tdg = td.nuc_grad_method()
        g1 = tdg.kernel(td.xy[2])
        self.assertAlmostEqual(g1[0,2], -0.40279473514282405, 6)

        td_solver = td.as_scanner()
        e1 = td_solver(pmol.set_geom_('H 0 0 1.805; F 0 0 0', unit='B'))
        e2 = td_solver(pmol.set_geom_('H 0 0 1.803; F 0 0 0', unit='B'))
        self.assertAlmostEqual((e1[2]-e2[2])/.002, g1[0,2], 5)

    def test_tda_b88(self):
        td = tdscf.TDA(mf_gga).run(nstates=3)
        tdg = td.nuc_grad_method()
        g1 = tdg.kernel(state=3)
        self.assertAlmostEqual(g1[0,2], -0.8120037135120326, 6)

        td_solver = td.as_scanner()
        e1 = td_solver(pmol.set_geom_('H 0 0 1.805; F 0 0 0', unit='B'))
        e2 = td_solver(pmol.set_geom_('H 0 0 1.803; F 0 0 0', unit='B'))
        self.assertAlmostEqual((e1[2]-e2[2])/.002, g1[0,2], 5)

    def test_tddft_lda(self):
        td = tdscf.TDDFT(mf_lda).run(nstates=3)
        tdg = td.nuc_grad_method()
        g1 = tdg.kernel(state=3)
        self.assertAlmostEqual(g1[0,2], -0.39791714992157035, 6)

        td_solver = td.as_scanner()
        e1 = td_solver(pmol.set_geom_('H 0 0 1.805; F 0 0 0', unit='B'))
        e2 = td_solver(pmol.set_geom_('H 0 0 1.803; F 0 0 0', unit='B'))
        self.assertAlmostEqual((e1[2]-e2[2])/.002, g1[0,2], 5)

    def test_tddft_b3lyp(self):
        mf = dft.UKS(mol).set(conv_tol=1e-12)
        mf.xc = '.2*HF + .8*b88, vwn'
        mf.scf()
        td = tdscf.TDDFT(mf).run(nstates=3)
        tdg = td.nuc_grad_method()
        g1 = tdg.kernel(state=3)
        self.assertAlmostEqual(g1[0,2], -0.80446691153291727, 6)

        td_solver = td.as_scanner()
        e1 = td_solver(pmol.set_geom_('H 0 0 1.805; F 0 0 0', unit='B'))
        e2 = td_solver(pmol.set_geom_('H 0 0 1.803; F 0 0 0', unit='B'))
        self.assertAlmostEqual((e1[2]-e2[2])/.002, g1[0,2], 4)

    def test_range_separated(self):
        mol = gto.M(atom="H; H 1 1.", basis='631g', verbose=0)
        mf = dft.UKS(mol).set(xc='CAMB3LYP')
        mf._numint.libxc = dft.xcfun
        td = mf.apply(tdscf.TDA)
        tdg_scanner = td.nuc_grad_method().as_scanner()
        g = tdg_scanner(mol, state=3)[1]
        self.assertAlmostEqual(lib.finger(g), -0.46656653988919661, 6)
        smf = td.as_scanner()
        e1 = smf(mol.set_geom_("H; H 1 1.001"))[2]
        e2 = smf(mol.set_geom_("H; H 1 0.999"))[2]
        self.assertAlmostEqual((e1-e2)/0.002*lib.param.BOHR, g[1,0], 4)


if __name__ == "__main__":
    print("Full Tests for TD-UKS gradients")
    unittest.main()

