#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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

import unittest
import tempfile
from functools import reduce
import numpy
from pyscf import gto
from pyscf import lib
from pyscf.gto.basis import parse_molpro
from pyscf.gto.basis import parse_gaussian
from pyscf.lib.exceptions import BasisNotFoundError

class KnownValues(unittest.TestCase):
    def test_parse_pople(self):
        self.assertEqual(gto.basis._parse_pople_basis('631g(d)', 'C'),
                         ('pople-basis/6-31G.dat', 'pople-basis/6-31G-polarization-d.dat'))
        self.assertEqual(gto.basis._parse_pople_basis('631g**', 'C'),
                         ('pople-basis/6-31Gss.dat',))
        self.assertEqual(gto.basis._parse_pople_basis('631++g**', 'C'),
                         ('pople-basis/6-31++Gss.dat',))
        self.assertEqual(gto.basis._parse_pople_basis('6311+g(d,p)', 'C'),
                         ('pople-basis/6-311+G.dat', 'pople-basis/6-311G-polarization-d.dat'))
        self.assertRaises(KeyError, gto.basis._parse_pople_basis, '631g++', 'C')

    def test_basis_load(self):
        self.assertEqual(gto.basis.load(__file__, 'H'), [])
        self.assertRaises(BasisNotFoundError, gto.basis.load, 'abas', 'H')
        #self.assertRaises(BasisNotFoundError, gto.basis.load(__file__, 'C'), [])

        self.assertEqual(len(gto.basis.load('631++g**', 'C')), 8)
        self.assertEqual(len(gto.basis.load('ccpcvdz', 'C')), 7)

        basdat = gto.basis.load('minao', 'C') + gto.basis.load('sto3g', 'C')
        basdat1 = gto.basis.parse_nwchem.parse(
            gto.basis.parse_nwchem.convert_basis_to_nwchem('C', basdat), 'C')
        bas = []
        for b in sorted(basdat, reverse=True):
            b1 = b[:1]
            for x in b[1:]:
                b1.append(list(x))
            bas.append(b1)
        bas = [b for b in bas if b[0]==0] + [b for b in bas if b[0]==1]
        self.assertEqual(bas, basdat1)

        self.assertEqual(len(gto.basis.load('def2-svp', 'Rn')), 16)

    def test_basis_load_from_file(self):
        ftmp = tempfile.NamedTemporaryFile()
        ftmp.write('''
Li    S
     16.1195750              0.15432897
      2.9362007              0.53532814
      0.7946505              0.44463454
Li    S
      0.6362897             -0.09996723
      0.1478601              0.39951283
      0.0480887              0.70011547
                   '''.encode())
        ftmp.flush()
        b = gto.basis.load(ftmp.name, 'Li')
        self.assertEqual(len(b), 2)
        self.assertEqual(len(b[0][1:]), 3)
        self.assertEqual(len(b[1][1:]), 3)

    def test_basis_load_ecp(self):
        self.assertEqual(gto.basis.load_ecp(__file__, 'H'), [])

    def test_parse_basis(self):
        basis_str = '''
#BASIS SET: (6s,3p) -> [2s,1p]
C    S
     71.6168370              0.15432897       
     13.0450960              0.53532814       
#
      3.5305122              0.44463454       
C    SP
      2.9412494             -0.09996723             0.15591627       
      0.6834831              0.39951283             0.60768372       
      0.2222899              0.70011547             0.39195739       '''
        self.assertRaises(BasisNotFoundError, gto.basis.parse_nwchem.parse, basis_str, 'O')
        basis_dat = gto.basis.parse_nwchem.parse(basis_str)
        self.assertEqual(len(basis_dat), 3)

    def test_parse_ecp(self):
        ecp_str = '''
#
Na nelec 10
Na ul
1    175.5502590            -10.0000000        
2     35.0516791            -47.4902024        
#
2      7.9060270            -17.2283007        

Na S
0    243.3605846              3.0000000        
1     41.5764759             36.2847626        
2     13.2649167             72.9304880        
Na P
0   1257.2650682              5.0000000        
1    189.6248810            117.4495683        
2     54.5247759            423.3986704        
'''
        ecpdat = gto.basis.parse_nwchem.parse_ecp(ecp_str, 'Na')
        self.assertEqual(ecpdat[0], 10)
        self.assertEqual(len(ecpdat[1]), 3)
        ecpdat1 = gto.basis.parse_nwchem.parse_ecp(ecp_str)
        self.assertEqual(ecpdat, ecpdat1)

        ecpdat1 = gto.basis.parse_nwchem.parse_ecp(
            gto.basis.parse_nwchem.convert_ecp_to_nwchem('Na', ecpdat), 'Na')
        self.assertEqual(ecpdat, ecpdat1)

    def test_optimize_contraction(self):
        bas = gto.parse(r'''
#BASIS SET: (6s,3p) -> [2s,1p]
        C    S
              2.9412494             -0.09996723
              0.6834831              0.39951283
              0.2222899              0.70011547
        C    S
              2.9412494             0.15591627
              0.6834831             0.60768372
              0.2222899             0.39195739
                                    ''', optimize=True)
        self.assertEqual(len(bas), 1)

        bas = [[1, 0,
                [2.9412494, -0.09996723],
                [0.6834831,  0.39951283],
                [0.2222899,  0.70011547]],
               [1, 1,
                [2.9412494, -0.09996723],
                [0.6834831,  0.39951283],
                [0.2222899,  0.70011547]],
               [1, 1,
                [2.9412494,  0.15591627],
                [0.6834831,  0.60768372],
                [0.2222899,  0.39195739]]]
        bas = gto.basis.parse_nwchem.optimize_contraction(bas)
        self.assertEqual(len(bas), 2)

    def test_remove_zero(self):
        bas = gto.parse(r'''
        C    S
        7.2610457926   0.0000000000   0.0000000000
        2.1056583087   0.0000000000   0.0000000000
        0.6439906571   1.0000000000   0.0000000000
        0.0797152017   0.0000000000   1.0000000000
        0.0294029590   0.0000000000   0.0000000000
                                    ''')
        self.assertEqual(len(bas[0]), 3)

        bas = [[0, 0,
                [7.2610457926,  0.0000000000,  0.0000000000],
                [2.1056583087,  0.0000000000,  0.0000000000],
                [0.6439906571,  1.0000000000,  0.0000000000],
                [0.0797152017,  0.0000000000,  1.0000000000],
                [0.0294029590,  0.0000000000,  0.0000000000]]]
        bas = gto.basis.parse_nwchem.remove_zero(bas)
        self.assertEqual(len(bas[0]), 4)

    def test_parse_molpro_basis(self):
        basis_str = '''
C s aug-cc-pVTZ AVTZ : 11 5 1.10 1.10 8.8 10.10 11.11
aug-cc-pVTZ
8236 1235 280.8 79.27 25.59 8.997 3.319 0.9059 0.3643 0.1285 0.04402
0.000531 0.004108 0.021087 0.081853 0.234817 0.434401 0.346129 0.039378
-0.008983 0.002385 -0.000113 -0.000878 -0.00454 -0.018133 -0.05576
-0.126895 -0.170352 0.140382 0.598684 0.395389 1 1 1
C p aug-cc-pVTZ AVTZ : 6 4 1.5 4.4 5.5 6.6
aug-cc-pVTZ
18.71 4.133 1.2 0.3827 0.1209 0.03569 0.014031 0.086866 0.290216
0.501008 0.343406 1 1 1
C d aug-cc-pVTZ AVTZ : 3 0
aug-cc-pVTZ
1.097 0.318 0.1
C f aug-cc-pVTZ AVTZ : 2 0
aug-cc-pVTZ
0.761 0.268
'''
        basis1 = parse_molpro.parse(basis_str)
        ref = gto.basis.parse('''
#BASIS SET: (11s,6p,3d,2f) -> [5s,4p,3d,2f]
C    S
   8236.0000000              0.0005310             -0.0001130              0.0000000              0.0000000        0
   1235.0000000              0.0041080             -0.0008780              0.0000000              0.0000000        0
    280.8000000              0.0210870             -0.0045400              0.0000000              0.0000000        0
     79.2700000              0.0818530             -0.0181330              0.0000000              0.0000000        0
     25.5900000              0.2348170             -0.0557600              0.0000000              0.0000000        0
      8.9970000              0.4344010             -0.1268950              0.0000000              0.0000000        0
      3.3190000              0.3461290             -0.1703520              0.0000000              0.0000000        0
      0.9059000              0.0393780              0.1403820              1.0000000              0.0000000        0
      0.3643000             -0.0089830              0.5986840              0.0000000              0.0000000        0
      0.1285000              0.0023850              0.3953890              0.0000000              1.0000000        0
      0.0440200              0.0000000              0.0000000              0.0000000              0.0000000        1.0000000        
C    P
     18.7100000              0.0140310              0.0000000              0.0000000          0
      4.1330000              0.0868660              0.0000000              0.0000000          0
      1.2000000              0.2902160              0.0000000              0.0000000          0
      0.3827000              0.5010080              1.0000000              0.0000000          0
      0.1209000              0.3434060              0.0000000              1.0000000          0
      0.0356900              0.0000000              0.0000000              0.0000000          1.0000000        
C    D
      1.0970000              1.0000000
C    D
      0.3180000              1.0000000        
C    D
      0.1000000              1.0000000        
C    F
      0.7610000              1.0000000        
C    F
      0.2680000              1.0000000        
END''')
        self.assertEqual(ref, basis1)

        basis_str = '''
c s 631g sv : 10 3 1.6 7.9 10.10
  3047.52500d+00  457.369500d+00  103.948700d+00  29.2101600d+00  9.28666300d+00
  3.16392700d+00  7.86827200d+00  1.88128900d+00  0.54424930d+00  0.16871440d+00
  1.83473700d-03  1.40373200d-02  0.06884262d+00  0.23218444d+00  0.46794130d+00
  0.36231200d+00 -0.11933240d+00 -0.16085420d+00  1.14345600d+00  1.00000000d+00
c p 631g sv : 4 2 1.3 4.4
  7.86827200d+00  1.88128900d+00  0.54424930d+00  0.16871440d+00  0.06899907d+00
  0.31642340d+00  0.74430830d+00  1.00000000d+00
'''
        basis1 = parse_molpro.parse(basis_str)
        ref = gto.basis.parse('''
#BASIS SET: (10s,4p) -> [3s,2p]
C    S
   3047.5250000              0.001834737            0                0
    457.3695000              0.01403732             0                0
    103.9487000              0.06884262             0                0
     29.2101600              0.23218444             0                0
      9.2866630              0.4679413              0                0
      3.1639270              0.3623120              0                0
      7.8682720              0                     -0.1193324        0
      1.8812890              0                     -0.1608542        0
      0.5442493              0                      1.1434560        0
      0.1687144              0                      0.0000000        1
C    P
      7.8682720              0.06899907       0
      1.8812890              0.3164234        0
      0.5442493              0.7443083        0
      0.1687144              0                1
END ''')
        self.assertEqual(ref, basis1)

    def test_parse_gaussian_basis(self):
        basis_str = '''
****
C     0
S   8   1.00
   8236.0000000              0.0005310        
   1235.0000000              0.0041080        
    280.8000000              0.0210870        
     79.2700000              0.0818530        
     25.5900000              0.2348170        
      8.9970000              0.4344010        
      3.3190000              0.3461290        
      0.3643000             -0.0089830        
S   8   1.00
   8236.0000000             -0.0001130        
   1235.0000000             -0.0008780        
    280.8000000             -0.0045400        
     79.2700000             -0.0181330        
     25.5900000             -0.0557600        
      8.9970000             -0.1268950        
      3.3190000             -0.1703520        
      0.3643000              0.5986840        
S   1   1.00
      0.9059000              1.0000000        
S   1   1.00
      0.1285000              1.0000000        
S   1   1.00
      0.0440200              1.0000000        
P   3   1.00
     18.7100000              0.0140310        
      4.1330000              0.0868660        
      1.2000000              0.2902160        
P   1   1.00
      0.3827000              1.0000000        
P   1   1.00
      0.1209000              1.0000000        
P   1   1.00
      0.0356900              1.0000000        
D   1   1.00
      1.0970000              1.0000000        
D   1   1.00
      0.3180000              1.0000000        
D   1   1.00
      0.1000000              1.0000000        
F   1   1.00
      0.7610000              1.0000000        
F   1   1.00
      0.2680000              1.0000000        
****
        '''
        basis1 = parse_gaussian.parse(basis_str)
        ref = gto.basis.load('augccpvtz', 'C')
        self.assertEqual(ref, basis1)

        basis_str = '''
****
C     0
S   6   1.00
   4563.2400000              0.00196665       
    682.0240000              0.0152306        
    154.9730000              0.0761269        
     44.4553000              0.2608010        
     13.0290000              0.6164620        
      1.8277300              0.2210060        
SP   3   1.00
     20.9642000              0.1146600              0.0402487        
      4.8033100              0.9199990              0.2375940        
      1.4593300             -0.00303068             0.8158540        
SP   1   1.00
      0.4834560              1.0000000              1.0000000        
SP   1   1.00
      0.1455850              1.0000000              1.0000000        
SP   1   1.00
      0.0438000              1.0000000              1.0000000        
D   1   1.00
      0.6260000              1.0000000        
****
'''
        basis1 = parse_gaussian.parse(basis_str)
        ref = gto.basis.load('6311++g*', 'C')
        self.assertEqual(ref, basis1)

        basis_str = '''
****
C     0 
S   6   1.00
   3047.5249000              0.0018347        
    457.3695100              0.0140373        
    103.9486900              0.0688426        
     29.2101550              0.2321844        
      9.2866630              0.4679413        
      3.1639270              0.3623120        
SP   3   1.00
      7.8682724             -0.1193324              0.0689991        
      1.8812885             -0.1608542              0.3164240        
      0.5442493              1.1434564              0.7443083        
SP   1   1.00
      0.1687144              1.0000000              1.0000000        
D   1   1.00
      2.5040000              1.0000000        
D   1   1.00
      0.6260000              1.0000000        
D   1   1.00
      0.1565000              1.0000000        
F   1   1.00
      0.8000000              1.0000000        
****
'''
        basis1 = parse_gaussian.parse(basis_str)
        ref = gto.basis.load('631g(3df,3pd)', 'C')
        self.assertEqual(ref, basis1)

    def test_parse_gaussian_load_basis(self):
        with tempfile.NamedTemporaryFile(mode='w+') as f:
            f.write('''
****
H 0
S 1 1.0
1.0 1.0
****
''')
            f.flush()
            self.assertEqual(parse_gaussian.load(f.name, 'H'), [[0, [1., 1.]]])

        with tempfile.NamedTemporaryFile(mode='w+') as f:
            f.write('''
H 0
S 1 1.0
1.0 1.0
****
''')
            f.flush()
            self.assertEqual(parse_gaussian.load(f.name, 'H'), [[0, [1., 1.]]])

        with tempfile.NamedTemporaryFile(mode='w+') as f:
            f.write('''
****
H 0
S 1 1.0
1.0 1.0
''')
            f.flush()
            self.assertEqual(parse_gaussian.load(f.name, 'H'), [[0, [1., 1.]]])

        with tempfile.NamedTemporaryFile(mode='w+') as f:
            f.write('''
H 0
S 1 1.0
1.0 1.0
''')
            f.flush()
            self.assertEqual(parse_gaussian.load(f.name, 'H'), [[0, [1., 1.]]])

    def test_basis_truncation(self):
        b = gto.basis.load('ano@3s1p1f', 'C')
        self.assertEqual(len(b), 3)
        self.assertEqual(len(b[0][1]), 4)
        self.assertEqual(len(b[1][1]), 2)
        self.assertEqual(b[2][0], 3)
        self.assertEqual(len(b[2][1]), 2)

        b = gto.basis.load('631g(3df,3pd)@3s2p1f', 'C')
        self.assertEqual(len(b), 6)
        self.assertEqual(len(b[0][1]), 2)
        self.assertEqual(len(b[1][1]), 2)
        self.assertEqual(len(b[2][1]), 2)
        self.assertEqual(len(b[3][1]), 2)
        self.assertEqual(len(b[4][1]), 2)
        self.assertEqual(b[5][0], 3)
        self.assertEqual(len(b[5][1]), 2)

        b = gto.basis.load('aug-ccpvtz@4s3p', 'C')
        self.assertEqual(len(b), 6)
        self.assertEqual(b[3][0], 1)

        self.assertRaises(AssertionError, gto.basis.load, 'aug-ccpvtz@4s3f', 'C')

    def test_to_general_contraction(self):
        b = gto.basis.to_general_contraction(gto.load('cc-pvtz', 'H'))
        self.assertEqual(len(b), 3)
        self.assertEqual(len(b[0]), 6)
        self.assertEqual(len(b[1]), 3)
        self.assertEqual(len(b[2]), 2)

if __name__ == "__main__":
    print("test basis module")
    unittest.main()
