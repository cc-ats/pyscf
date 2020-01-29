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
# Authors: Timothy Berkelbach <tim.berkelbach@gmail.com>
#          Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy as np
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import dft as pbcdft
import pyscf.pbc
pyscf.pbc.DEBUG = False

L = 4.
cell = pbcgto.Cell()
cell.verbose = 0
cell.a = np.eye(3)*L
cell.atom =[['He' , ( L/2+0., L/2+0. ,   L/2+1.)],]
cell.basis = {'He': [[0, (4.0, 1.0)], [0, (1.0, 1.0)]]}
cell.build()

def tearDownModule():
    global cell
    del cell


class KnownValues(unittest.TestCase):
#    def test_lda_grid30(self):
#        cell = pbcgto.Cell()
#        cell.unit = 'B'
#        L = 10
#        cell.a = np.diag([L]*3)
#        cell.mesh = np.array([41]*3)
#        cell.atom = [['He', (L/2.,L/2.,L/2.)], ]
## these are some exponents which are not hard to integrate
#        cell.basis = { 'He': [[0, (0.8, 1.0)],
#                              [0, (1.0, 1.0)],
#                              [0, (1.2, 1.0)]] }
#        cell.verbose = 5
#        cell.output = '/dev/null'
#        cell.pseudo = None
#        cell.build()
#        mf = pbcdft.RKS(cell)
#        mf.xc = 'LDA,VWN_RPA'
#        mf.kpt = np.ones(3)
#        e1 = mf.scf()
#        self.assertAlmostEqual(e1, -2.6409616064015591, 8)
#
#
#    def test_pp_RKS(self):
#        cell = pbcgto.Cell()
#
#        cell.unit = 'A'
#        cell.atom = '''
#            Si    0.000000000    0.000000000    0.000000000;
#            Si    0.000000000    2.715348700    2.715348700;
#            Si    2.715348700    2.715348700    0.000000000;
#            Si    2.715348700    0.000000000    2.715348700;
#            Si    4.073023100    1.357674400    4.073023100;
#            Si    1.357674400    1.357674400    1.357674400;
#            Si    1.357674400    4.073023100    4.073023100;
#            Si    4.073023100    4.073023100    1.357674400
#        '''
#        cell.basis = 'gth-szv'
#        cell.pseudo = 'gth-pade'
#
#        Lx = Ly = Lz = 5.430697500
#        cell.a = np.diag([Lx,Ly,Lz])
#        cell.mesh = np.array([21]*3)
#
#        cell.verbose = 5
#        cell.output = '/dev/null'
#        cell.build()
#
#        mf = pbcdft.RKS(cell)
#        mf.xc = 'lda,vwn'
#        self.assertAlmostEqual(mf.scf(), -31.081616722101646, 8)


    def test_chkfile_k_point(self):
        cell = pbcgto.Cell()
        cell.a = np.eye(3) * 6
        cell.mesh = [21]*3
        cell.unit = 'B'
        cell.atom = '''He     2.    2.       3.
                      He     3.    2.       3.'''
        cell.basis = {'He': 'sto3g'}
        cell.verbose = 0
        cell.build()
        mf1 = pbcdft.RKS(cell)
        mf1.max_cycle = 1
        mf1.kernel()

        cell = pbcgto.Cell()
        cell.a = np.eye(3) * 6
        cell.mesh = [41]*3
        cell.unit = 'B'
        cell.atom = '''He     2.    2.       3.
                       He     3.    2.       3.'''
        cell.basis = {'He': 'ccpvdz'}
        cell.verbose = 5
        cell.output = '/dev/null'
        cell.nimgs = [2,2,2]
        cell.build()
        mf = pbcdft.RKS(cell)
        np.random.seed(10)
        mf.kpt = np.random.random(3)
        mf.max_cycle = 1
        dm = mf.from_chk(mf1.chkfile)
        mf.conv_check = False
        self.assertAlmostEqual(mf.scf(dm), -4.7090816314173365, 8)

    def test_density_fit(self):
        L = 4.
        cell = pbcgto.Cell()
        cell.a = np.eye(3)*L
        cell.atom =[['He' , ( L/2+0., L/2+0. ,   L/2+1.)],
                    ['He' , ( L/2+1., L/2+0. ,   L/2+1.)]]
        cell.basis = {'He': [[0, (4.0, 1.0)], [0, (1.0, 1.0)]]}
        cell.build()
        mf = pbcdft.RKS(cell).density_fit()
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -4.717699891018736, 7)

    def test_rsh_fft(self):
        mf = pbcdft.RKS(cell)
        mf.xc = 'camb3lyp'
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -2.3032261128220544, 7)

    def test_rsh_df(self):
        mf = pbcdft.RKS(cell).density_fit()
        mf.xc = 'camb3lyp'
        mf.omega = .15
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -2.399571378419408, 7)

    def test_rsh_mdf(self):
        mf = pbcdft.RKS(cell).mix_density_fit()
        mf.xc = 'camb3lyp'
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -2.303225896642264, 7)

    def test_rsh_aft_high_cost(self):
        from pyscf.pbc.df.aft import AFTDF
        mf = pbcdft.RKS(cell)
        mf.with_df = AFTDF(cell)
        mf.xc = 'camb3lyp'
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -2.303226113014942, 7)

    def test_rsh_0d(self):
        L = 4.
        cell = pbcgto.Cell()
        cell.verbose = 0
        cell.a = np.eye(3)*L
        cell.atom =[['He' , ( L/2+0., L/2+0. ,   L/2+1.)],]
        cell.basis = {'He': [[0, (4.0, 1.0)], [0, (1.0, 1.0)]]}
        cell.dimension = 0
        cell.mesh = [60]*3
        cell.build()
        mf = pbcdft.RKS(cell).density_fit()
        mf.xc = 'camb3lyp'
        mf.omega = '0.7'
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -2.4836476875900297, 7)

        mol = cell.to_mol()
        mf1 = mol.RKS().density_fit()
        mf1.xc = 'camb3lyp'
        mf1.omega = '0.7'
        mf1.kernel()
        self.assertAlmostEqual(mf1.e_tot, mf.e_tot, 4)

if __name__ == '__main__':
    print("Full Tests for pbc.dft.rks")
    unittest.main()
