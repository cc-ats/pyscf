# Author: Junjie Yang <yangjunjie0320@gmail.com> Zheng Pei and Yihan Shao

import time
import copy
from functools import reduce
import numpy

from pyscf import lib
from pyscf.lib import logger
from pyscf import dft
from pyscf.dft import rks
from pyscf.dft import numint
from pyscf.scf import cphf

from pyscf.grad import rks as rks_grad
from pyscf.grad import tdrhf

def calc_relaxed_z(tdscf, x_y, singlet=True, atmlst=None,
           max_memory=2000, verbose=logger.INFO):
    from pyscf.grad.tdrks import _contract_xc_kernel

    log = logger.new_logger(tdscf, verbose)
    time0 = time.clock(), time.time()

    mol = tdscf.mol
    mf = tdscf._scf
    td_grad = tdscf.Gradients()

    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    nocc = (mo_occ>0).sum()
    nvir = nmo - nocc
    x, y = x_y
    xpy = (x+y).reshape(nocc,nvir).T
    xmy = (x-y).reshape(nocc,nvir).T
    orbv = mo_coeff[:,nocc:]
    orbo = mo_coeff[:,:nocc]

    dvv =  numpy.einsum('ai,bi->ab', xpy, xpy) + numpy.einsum('ai,bi->ab', xmy, xmy)
    doo = -numpy.einsum('ai,aj->ij', xpy, xpy) - numpy.einsum('ai,aj->ij', xmy, xmy)
    dmzvop = reduce(numpy.dot, (orbv, xpy, orbo.T))
    dmzvom = reduce(numpy.dot, (orbv, xmy, orbo.T))
    dmzoo = reduce(numpy.dot, (orbo, doo, orbo.T))
    dmzoo+= reduce(numpy.dot, (orbv, dvv, orbv.T))

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, td_grad.max_memory*.9-mem_now)

    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    # dm0 = mf.make_rdm1(mo_coeff, mo_occ), but it is not used when computing
    # fxc since rho0 is passed to fxc function.
    dm0 = None
    rho0, vxc, fxc = ni.cache_xc_kernel(mf.mol, mf.grids, mf.xc,
                                        [mo_coeff]*2, [mo_occ*.5]*2, spin=1)
    f1vo, f1oo, vxc1, k1ao = \
            _contract_xc_kernel(td_grad, mf.xc, dmzvop,
                                dmzoo, True, True, singlet, max_memory)

    if abs(hyb) > 1e-10:
        dm = (dmzoo, dmzvop+dmzvop.T, dmzvom-dmzvom.T)
        vj, vk = mf.get_jk(mol, dm, hermi=0)
        vk *= hyb
        if abs(omega) > 1e-10:
            vk += rks._get_k_lr(mol, dm, omega) * (alpha-hyb)
        veff0doo = vj[0] * 2 - vk[0] + f1oo[0] + k1ao[0] * 2
        wvo = reduce(numpy.dot, (orbv.T, veff0doo, orbo)) * 2
        if singlet:
            veff = vj[1] * 2 - vk[1] + f1vo[0] * 2
        else:
            veff = -vk[1] + f1vo[0] * 2
        veff0mop = reduce(numpy.dot, (mo_coeff.T, veff, mo_coeff))
        wvo -= numpy.einsum('ki,ai->ak', veff0mop[:nocc,:nocc], xpy) * 2
        wvo += numpy.einsum('ac,ai->ci', veff0mop[nocc:,nocc:], xpy) * 2
        veff = -vk[2]
        veff0mom = reduce(numpy.dot, (mo_coeff.T, veff, mo_coeff))
        wvo -= numpy.einsum('ki,ai->ak', veff0mom[:nocc,:nocc], xmy) * 2
        wvo += numpy.einsum('ac,ai->ci', veff0mom[nocc:,nocc:], xmy) * 2
    else:
        vj = mf.get_j(mol, (dmzoo, dmzvop+dmzvop.T), hermi=1)
        veff0doo = vj[0] * 2 + f1oo[0] + k1ao[0] * 2
        wvo = reduce(numpy.dot, (orbv.T, veff0doo, orbo)) * 2
        if singlet:
            veff = vj[1] * 2 + f1vo[0] * 2
        else:
            veff = f1vo[0] * 2
        veff0mop = reduce(numpy.dot, (mo_coeff.T, veff, mo_coeff))
        wvo -= numpy.einsum('ki,ai->ak', veff0mop[:nocc,:nocc], xpy) * 2
        wvo += numpy.einsum('ac,ai->ci', veff0mop[nocc:,nocc:], xpy) * 2
        veff0mom = numpy.zeros((nmo,nmo))
    def fvind(x):
# Cannot make call to .base.get_vind because first order orbitals are solved
# through closed shell ground state CPHF.
        dm = reduce(numpy.dot, (orbv, x.reshape(nvir,nocc), orbo.T))
        dm = dm + dm.T
# Call singlet XC kernel contraction, for closed shell ground state
        vindxc = numint.nr_rks_fxc_st(ni, mol, mf.grids, mf.xc, dm0, dm, 0,
                                      singlet, rho0, vxc, fxc, max_memory)
        if abs(hyb) > 1e-10:
            vj, vk = mf.get_jk(mol, dm)
            veff = vj * 2 - hyb * vk + vindxc
            if abs(omega) > 1e-10:
                veff -= rks._get_k_lr(mol, dm, omega, hermi=1) * (alpha-hyb)
        else:
            vj = mf.get_j(mol, dm)
            veff = vj * 2 + vindxc
        return reduce(numpy.dot, (orbv.T, veff, orbo)).ravel()
    z1 = cphf.solve(fvind, mo_energy, mo_occ, wvo,
                    max_cycle=td_grad.cphf_max_cycle, tol=td_grad.cphf_conv_tol)[0]
    z1 = z1.reshape(nvir,nocc)
    time1 = log.timer('Z-vector using CPHF solver', *time0)

    z1ao  = reduce(numpy.dot, (orbv, z1, orbo.T))
    return z1ao

def calc_trans_dm(tdscf, x_y):
    mf = tdscf._scf
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    nocc = (mo_occ>0).sum()
    x, y = x_y
    orbv = mo_coeff[:,nocc:]
    orbo = mo_coeff[:,:nocc]
    trans_dm  = numpy.einsum('ia,pi,qa->pq', x, orbo.conj(), orbv)
    trans_dm += numpy.einsum('ia,pi,qa->pq', y, orbo, orbv.conj())
    return trans_dm + trans_dm.T

def calc_diff_dm(tdscf, x_y):
    mf = tdscf._scf
    mo_coeff = mf.mo_coeff
    nao, nmo = mo_coeff.shape
    mo_occ = mf.mo_occ
    nocc = (mo_occ>0).sum()
    x, y = x_y
    orbv = mo_coeff[:,nocc:]
    orbo = mo_coeff[:,:nocc]

    diff_dm = numpy.zeros((nmo, nmo))
    diff_dm[nocc, :nocc]  -= numpy.einsum('ia,ja->ij', x, x)
    diff_dm[nocc, :nocc]  -= numpy.einsum('ia,ja->ij', y, y)
    diff_dm[nocc:, nocc:] += numpy.einsum('ia,ib->ab', x, x)
    diff_dm[nocc:, nocc:] += numpy.einsum('ia,ib->ab', y, y)
    # print("self.diff_dm\n", self.diff_dm)
    diff_dm = numpy.einsum('ij,pi,qj->pq', diff_dm, mo_coeff, mo_coeff.conj())
    diff_dm *= 2.0
    return diff_dm

def proj_ex_states(tdscf, dm_ao):
    mol = tdscf.mol
    mf = tdscf._scf
    mo_coeff = mf.mo_coeff

    if (dm_ao.ndim == 2):
        mo_occ = mf.mo_occ
        nao, nmo = mo_coeff.shape
        nocc = (mo_occ>0).sum()
        nvir = nmo - nocc

        x = mf.mo_coeff
        x_inv = numpy.einsum('li,ls->is', x, mf.get_ovlp())
        x_t_inv = x_inv.T
        dm_mo = reduce(numpy.dot, (x_inv, dm_ao, x_t_inv))
        dm_mo_ov = dm_mo[:nocc, nocc:].reshape(nocc,nvir).T
        xmy = [(tdscf.xy[i][0]-tdscf.xy[i][1]).reshape(nocc,nvir).T for i in range(len(tdscf.xy))]
        proj = 2*numpy.einsum("ijk,jk->i",xmy, dm_mo_ov)
        return proj

    elif (dm_ao.ndim == 3 and dm_ao.shape[0] == 2):
        mo_occ_a, mo_occ_b = mf.mo_occ
        nao, nmo = mo_coeff[0].shape
        nocc_a = (mo_occ_a>0).sum()
        nocc_b = (mo_occ_b>0).sum()
        nvir_a = nmo - nocc_a
        nvir_b = nmo - nocc_b

        x = mf.mo_coeff
        x_t = numpy.einsum('aij->aji', x)
        x_inv = numpy.einsum('ali,ls->ais', x, mf.get_ovlp())
        x_t_inv = numpy.einsum('aij->aji', x_inv)
        dm_mo_a = reduce(numpy.dot, (x_inv[0], dm_ao[0], x_t_inv[0]))
        dm_mo_b = reduce(numpy.dot, (x_inv[1], dm_ao[1], x_t_inv[1]))
        dm_mo_a_ov = dm_mo_a[:nocc_a, nocc_a:].reshape(nocc_a,nvir_a).T
        dm_mo_b_ov = dm_mo_b[:nocc_b, nocc_b:].reshape(nocc_b,nvir_b).T
        
        xmy_a = [(tdscf.xy[0][i][0]-tdscf.xy[0][i][1]).reshape(nocc_a,nvir_a).T for i in range(len(tdscf.xy[0]))]
        xmy_b = [(tdscf.xy[1][i][0]-tdscf.xy[1][i][1]).reshape(nocc_a,nvir_a).T for i in range(len(tdscf.xy[1]))]
        xmy_b = [(tdscf.xy[2][i][0]-tdscf.xy[2][i][1]).reshape(nocc_a,nvir_a).T for i in range(len(tdscf.xy[2]))]
        print("tdscf.xy[0].shape = ", tdscf.xy[0].shape)
        print("tdscf.xy[1].shape = ", tdscf.xy[1].shape)
        print("tdscf.xy[2].shape = ", tdscf.xy[2].shape)
        assert  1==2
        xmy_a = [(tdscf.xy[i][0]-tdscf.xy[i][1]).reshape(nocc,nvir).T for i in range(len(tdscf.xy))]




if __name__ == "__main__":
    from pyscf import gto, scf, dft, tddft

    mol = gto.Mole()
    mol.atom = '''
    C         0.6584188440    0.0000000000    0.0000000000
    C        -0.6584188440    0.0000000000    0.0000000000
    H         1.2256624142    0.9143693597    0.0000000000
    H         1.2256624142   -0.9143693597    0.0000000000
    H        -1.2256624142    0.9143693597    0.0000000000
    H        -1.2256624142   -0.9143693597    0.0000000000'''
    mol.basis = '6-31g(d)'
    mol.build()

    mf1 = dft.UKS(mol)
    mf1.xc = "PBE"
    mf1.max_cycle = 100
    mf1.conv_tol = 1e-12
    mf1.verbose = 4
    mf1.kernel()
    dm1 = mf1.make_rdm1()
    
    mf2 = dft.UKS(mol)
    mf2.xc = "pbe"
    mf2.max_cycle = 200
    mf2.conv_tol = 1e-12
    mf2.verbose = 3
    h1e_0 = mf2.get_hcore()
    e  = -1e-3
    ee = [e, 0, 0]
    ao_dip = mf2.mol.intor_symmetric('int1e_r', comp=3)
    ef = numpy.einsum('x,xij->ij', ee, ao_dip )
    h1e = h1e_0 + ef
    mf2.get_hcore = lambda *args: h1e
    mf2.kernel(dm=dm1)
    dm2  = mf2.make_rdm1()

    td = tddft.TDDFT(mf1)
    td.nstates = 5
    td.kernel()
    am = proj_ex_states(td, dm2)
    print("am = \n", am)
