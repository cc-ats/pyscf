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

import numpy as np

#
#
#
def get_sp_mu2s(sp2nmult,sp_mu2j):
    """  Generates list of start indices for atomic orbitals, based on the counting arrays """
    sp_mu2s = []
    for sp,(nmu,mu2j) in enumerate(zip(sp2nmult,sp_mu2j)):
        mu2s = np.zeros((nmu+1), dtype='int64')
        for mu in range(nmu):
            mu2s[mu+1] = sum(2*mu2j[0:mu+1]+1)
        sp_mu2s.append(mu2s)
    
    return sp_mu2s
