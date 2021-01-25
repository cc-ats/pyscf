#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A, B matrices of TDDFT method.
'''

import numpy
from pyscf import gto, scf, dft, tdscf

def make_qed_tddft(td_obj, mode_list):
    freq_list, vec_list = mode_list
    num_mode = len(freq_list)
    assert len(freq_list) == len(vec_list)
    a, b   = td_obj.get_ab()
    nocc, nvir = a.shape[:2]
    dip_ao = td_obj.mol.intor("cint1e_r_sph", comp=3)
    dip_mo = numpy.einsum('mp,xmn,nq->xpq', td_obj._scf.mo_coeff, dip_ao, td_obj._scf.mo_coeff)
    dip_vo  = dip_mo[:,nocc:,:nocc]

    n_list = []
    m_list = []

    for alpha, lam_alpha in enumerate(vec_list):
        temp = numpy.einsum('x,xai,y,ybj->aibj', lam_alpha, dip_vo, lam_alpha, dip_vo).reshape(a.shape)
        a += temp
        b += temp
        n_alpha = - numpy.einsum('x,xai->ai', lam_alpha, dip_vo)/2
        n_list.append(n_alpha)
        m_alpha = - freq_list[alpha] * numpy.einsum('x,xai->ai', lam_alpha, dip_vo)
        m_list.append(m_alpha)

    a = a.reshape(nocc*nvir,nocc*nvir)
    b = b.reshape(nocc*nvir,nocc*nvir)

    n_array = numpy.array(n_list).reshape(num_mode, -1)
    m_array = numpy.array(m_list).reshape(-1, num_mode)

    omega_block = numpy.diag(freq_list)
    zero_block  = numpy.zeros_like(omega_block)

    e, v = numpy.linalg.eig(numpy.bmat([[a        ,        b       ,    m_array,         m_array          ],
                                        [-b.conj(),       -a.conj(),      -m_array.conj(), -m_array.conj()],
                                        [n_array,          n_array,        omega_block,     zero_block    ],
                                        [-n_array.conj(), -n_array.conj(), zero_block,     -omega_block   ]]
                                      ))
    nroots = 10
    index = numpy.argsort(e[e > 0])[:nroots]
    ee = e[e > 0][index]
    vv = v[:, e > 0][:, index]

    x_list = []
    y_list = []
    p_list = []
    q_list = []

    for i in range(nroots):
        n = 0.0

        x = vv[0:nocc*nvir, i].reshape(nvir,nocc)
        y = vv[nocc*nvir:2*nocc*nvir, i].reshape(nvir,nocc)
        n += numpy.einsum("ai,ai->", x, x) - numpy.einsum("ai,ai->", y, y)

        p = vv[2*nocc*nvir:(2*nocc*nvir+num_mode), i].reshape(-1)
        q = vv[(2*nocc*nvir+num_mode):(2*nocc*nvir+2*num_mode), i].reshape(-1)
        n += numpy.einsum("ip,ip->", p, p) - numpy.einsum("ip,ip->", q, q)

        sqrt_n = n/numpy.sqrt(numpy.abs(n))
        x_list.append(x/sqrt_n)
        y_list.append(y/sqrt_n)
        p_list.append(p/sqrt_n)
        q_list.append(q/sqrt_n)

    print(numpy.einsum("jai,jai->j", x_list, x_list) - numpy.einsum("jai,jai->j", y_list, y_list))
    print(numpy.einsum("jia,jia->j", p_list, p_list) - numpy.einsum("jia,jia->j", q_list, q_list))
    print(ee)

mol = gto.Mole()
mol.atom = '''
  H   -0.0000000    0.5547481    0.7830365
  O   -0.0000000   -0.0514344    0.0000000
  H    0.0000000    0.5547481   -0.7830365
'''
mol.basis = 'sto-3g'
mol.build()

mf = scf.RHF(mol).run()
td = tdscf.TDHF(mf)
td.verbose = 0
td.nroots  = 2
td.kernel()
dips = td.transition_dipole()

make_qed_tddft(td, ([0.4500828], [[0.00, 0.00, 0.001]]))