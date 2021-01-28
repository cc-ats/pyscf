#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A, B matrices of TDDFT method.
'''

import numpy
import scipy
from pyscf import gto, scf, dft, tdscf

def make_qed_test(td_obj, mode_list):
    freq_list, vec_list = mode_list
    num_mode = len(freq_list)
    assert len(freq_list) == len(vec_list)

    a0, b0     = td_obj.get_ab()
    nocc, nvir = a0.shape[:2]
    dip_ao = td_obj.mol.intor("cint1e_r_sph", comp=3)
    dip_mo = numpy.einsum('mp,xmn,nq->xpq', td_obj._scf.mo_coeff, dip_ao, td_obj._scf.mo_coeff)
    dip_vo = dip_mo[:,nocc:,:nocc]

    n_list = []
    m_list = []

    a = a0.reshape(nocc*nvir,nocc*nvir)
    b = b0.reshape(nocc*nvir,nocc*nvir)
    e0, v0 = numpy.linalg.eig(a)

    x0 = []
    for i in range(e0.shape[0]):
        x0.append(v0[:,i].reshape(nocc,nvir))

    for alpha, lam_alpha in enumerate(vec_list):
        temp = numpy.einsum('x,xai,y,ybj->iajb', lam_alpha, dip_vo, lam_alpha, dip_vo)
        a0 += temp

        n_alpha = -numpy.einsum('x,xai->ia', lam_alpha, dip_vo)/2
        m_alpha = -freq_list[alpha] * numpy.einsum('x,xai->ia', lam_alpha, dip_vo)

        n_list.append(n_alpha)
        m_list.append(m_alpha)

    n_new = numpy.einsum("nia,lia->ln", x0, n_list)
    m_new = numpy.einsum("nia,lia->nl", x0, m_list)

    e_block     = numpy.einsum("nia,iajb,mjb->nm", x0, a0, x0)
    omega_block = numpy.diag(freq_list)

    cis_mat = numpy.bmat([[e_block, m_new], [n_new, omega_block]])
    print(cis_mat)
    e, v = scipy.linalg.eig(cis_mat)

    nroots = 16
    index = numpy.argsort(e)[:nroots]
    ee = e[index]
    vv = v[:, index]

    x_list = []
    p_list = []

    for i in range(len(index)):
        print(vv[0:nocc*nvir,                    i])
        print(vv[nocc*nvir:(nocc*nvir+num_mode), i])

        x = vv[0:nocc*nvir,                    i].reshape(nvir,nocc)
        p = vv[nocc*nvir:(nocc*nvir+num_mode), i]

        x_list.append(x)
        p_list.append(p)

    elec_amp  = numpy.einsum("jai,jai->j", x_list, x_list)
    photo_amp = numpy.einsum("ji,ji->j", p_list, p_list)


    return ee, elec_amp, photo_amp


def make_qed_tda(td_obj, mode_list):
    freq_list, vec_list = mode_list
    num_mode = len(freq_list)
    assert len(freq_list) == len(vec_list)
    a, b       = td_obj.get_ab()
    nocc, nvir = a.shape[:2]
    dip_ao = td_obj.mol.intor("cint1e_r_sph", comp=3)
    dip_mo = numpy.einsum('mp,xmn,nq->xpq', td_obj._scf.mo_coeff, dip_ao, td_obj._scf.mo_coeff)
    dip_vo = dip_mo[:,nocc:,:nocc]

    n_list = []
    m_list = []

    for alpha, lam_alpha in enumerate(vec_list):
        temp = numpy.einsum('x,xai,y,ybj->iajb', lam_alpha, dip_vo, lam_alpha, dip_vo)

        a += temp
        b += temp

        n_alpha = -numpy.einsum('x,xai->ia', lam_alpha, dip_vo)/2
        m_alpha = -freq_list[alpha] * numpy.einsum('x,xai->ia', lam_alpha, dip_vo)

        n_list.append(n_alpha.reshape(nocc*nvir))
        m_list.append(m_alpha.reshape(nocc*nvir))

    a = a.reshape(nocc*nvir,nocc*nvir)
    b = b.reshape(nocc*nvir,nocc*nvir)

    n_array = numpy.array(n_list)
    m_array = numpy.array(m_list).T

    omega_block = numpy.diag(freq_list)
    zero_block  = numpy.zeros_like(omega_block)

    e0, v0 = numpy.linalg.eig(a)
    zero1 = numpy.zeros([num_mode, nocc*nvir])
    zero2 = numpy.eye(num_mode)
    zero3 = numpy.zeros([nocc*nvir,num_mode])
    v1     = numpy.bmat([[v0, zero3], [zero1, zero2]])
    cis_mat = numpy.bmat([[a, m_array], [n_array, omega_block]])
    cis_mat = numpy.einsum("pm,pq,qn->mn", v1, cis_mat, v1)
    
    e, v = scipy.linalg.eig(cis_mat)

    nroots = 16
    index = numpy.argsort(e)[:nroots]
    ee = e[index]
    vv = v[:, index]

    x_list = []
    p_list = []

    for i in range(len(index)):
        print(vv[0:nocc*nvir,                    i])
        print(vv[nocc*nvir:(nocc*nvir+num_mode), i])

        x = vv[0:nocc*nvir,                    i].reshape(nvir,nocc)
        p = vv[nocc*nvir:(nocc*nvir+num_mode), i]

        x_list.append(x)
        p_list.append(p)

    elec_amp  = numpy.einsum("jai,jai->j", x_list, x_list)
    photo_amp = numpy.einsum("ji,ji->j", p_list, p_list)
    return ee, elec_amp, photo_amp

def make_qed_rpa(td_obj, mode_list):
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

    e, v = numpy.linalg.eig(
        numpy.bmat([[a        ,        b       ,       m_array,         m_array       ],
                    [-b.conj(),       -a.conj(),      -m_array.conj(), -m_array.conj()],
                    [n_array,          n_array,        omega_block,     zero_block    ],
                    [-n_array.conj(), -n_array.conj(), zero_block,     -omega_block   ]])
                    )


    abba = numpy.bmat([[a        ,        b       ,       m_array,         m_array          ],
                                        [-b.conj(),       -a.conj(),      -m_array.conj(), -m_array.conj()],
                                        [n_array,          n_array,        omega_block,     zero_block    ],
                                        [-n_array.conj(), -n_array.conj(), zero_block,     -omega_block   ]]
                                      )
    numpy.savetxt("abba.csv", abba, delimiter=", ")

    nroots = 16
    index = numpy.argsort(e)[:nroots]
    ee = e[index]
    vv = v[:, index]

    x_list = []
    y_list = []
    p_list = []
    q_list = []

    for i in range(len(index)):
        print()
        n = 0.0
        print("e = ", ee[i])
        print("v = ", vv[:,i])
        x = vv[0:nocc*nvir, i].reshape(nvir,nocc)
        y = vv[nocc*nvir:2*nocc*nvir, i].reshape(nvir,nocc)
        n += numpy.einsum("ai,ai->", x.conj(), x) - numpy.einsum("ai,ai->", y, y)

        p = vv[2*nocc*nvir:(2*nocc*nvir+num_mode), i].reshape(-1)
        q = vv[(2*nocc*nvir+num_mode):(2*nocc*nvir+2*num_mode), i].reshape(-1)
        n += numpy.einsum("ip,ip->", p.conj(), p) - numpy.einsum("ip,ip->", q, q)

        print("n = ", n)
        sqrt_n = n/numpy.sqrt(numpy.abs(n))
        x_list.append(x/sqrt_n)
        print("x = ", x/sqrt_n)
        y_list.append(y/sqrt_n)
        print("y = ", y/sqrt_n)
        p_list.append(p/sqrt_n)
        print("p = ", p/sqrt_n)
        q_list.append(q/sqrt_n)
        print("q = ", q/sqrt_n)

    x_list = numpy.array(x_list)
    y_list = numpy.array(y_list)
    p_list = numpy.array(p_list)
    q_list = numpy.array(q_list)
    elec_amp  = numpy.einsum("jai,jai->j", x_list.conj(), x_list) - numpy.einsum("jai,jai->j", y_list.conj(), y_list)
    photo_amp = numpy.einsum("jia,jia->j", p_list.conj(), p_list) - numpy.einsum("jia,jia->j", q_list.conj(), q_list)
    return ee, elec_amp, photo_amp

mol = gto.Mole()
mol.atom = '''
  H    0.0000000    0.0000000    0.3540000
  H    0.0000000    0.0000000   -0.3540000
'''

mol.basis = '6-31g'
mol.build()

mf = dft.RKS(mol)
mf.xc = "LDA"
mf.kernel()
td = tdscf.TDDFT(mf)
td.verbose = 0
td.nroots  = 20
td.kernel()
dips = td.transition_dipole()

for d in [1e-3]:
    e, eamp, pamp = make_qed_test(td, 
    ([0.1, 1.63684123],[[0.00, 0.00, d], [0.00, 0.00, d]])
    )
    print("")
    fmt_str = "".join(["% .4f" for _ in e])
    print("d = %e"%d)
    print(e)
    print("ex    ene = " + fmt_str%tuple(e))
    print("elec  amp = " + fmt_str%tuple(eamp))
    print("photo amp = " + fmt_str%tuple(pamp))

    e, eamp, pamp = make_qed_tda(td, 
    ([0.1, 1.63684123],[[0.00, 0.00, d], [0.00, 0.00, d]])
    )
    print("")
    fmt_str = "".join(["% .4f" for _ in e])
    print("d = %e"%d)
    print(e)
    print("ex    ene = " + fmt_str%tuple(e))
    print("elec  amp = " + fmt_str%tuple(eamp))
    print("photo amp = " + fmt_str%tuple(pamp))