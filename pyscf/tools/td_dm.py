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
    mf  = tdscf._scf
    mol = tdscf.mol

    if isinstance(mf, scf.uhf.UHF):
        mf = scf.addons.convert_to_uhf(mf)
        assert (dm_ao.ndim == 3 and dm_ao.shape[0] == 2)
        mo_coeff_a, mo_coeff_b = mf.mo_coeff
        mo_occ_a, mo_occ_b = mf.mo_occ
        nao, nmo = mo_coeff_a.shape
        nocc_a = (mo_occ_a>0).sum()
        nocc_b = (mo_occ_b>0).sum()
        nvir_a = nmo - nocc_a
        nvir_b = nmo - nocc_b
        xmy_a = [(tdscf.xy[i][0][0]-tdscf.xy[i][1][0]).reshape(nocc_a,nvir_a).T for i in range(len(tdscf.xy))]
        xmy_b = [(tdscf.xy[i][0][1]-tdscf.xy[i][1][1]).reshape(nocc_b,nvir_b).T for i in range(len(tdscf.xy))]

        x_a = mo_coeff_a
        x_inv_a = numpy.einsum('li,ls->is', x_a, mf.get_ovlp())
        x_t_inv_a = x_inv_a.T
        dm_mo_a = reduce(numpy.dot, (x_inv_a, dm_ao[0], x_t_inv_a))
        dm_mo_ov_a = dm_mo_a[:nocc_a, nocc_a:].reshape(nocc_a,nvir_a).T

        x_b = mo_coeff_b
        x_inv_b = numpy.einsum('li,ls->is', x_b, mf.get_ovlp())
        x_t_inv_b = x_inv_b.T
        dm_mo_b = reduce(numpy.dot, (x_inv_b, dm_ao[1], x_t_inv_b))
        dm_mo_ov_b = dm_mo_b[:nocc_b, nocc_b:].reshape(nocc_b,nvir_b).T

        proj = numpy.einsum("ijk,jk->i",xmy_a, dm_mo_ov_a) + numpy.einsum("ijk,jk->i",xmy_b, dm_mo_ov_b)
        return 2*proj
    
    else:
        mf = scf.addons.convert_to_rhf(mf)
        assert dm_ao.ndim == 2
        mo_coeff = mf.mo_coeff
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

def eval_rt_dm(tdscf, dm_ao, am, e, t_array):
    mf  = tdscf._scf
    mol = tdscf.mol

    wmt = numpy.einsum("i,j->ij", e, t_array)

    if isinstance(mf, scf.uhf.UHF):
        mf = scf.addons.convert_to_uhf(mf)
        assert (dm_ao.ndim == 3 and dm_ao.shape[0] == 2)
        mo_coeff_a, mo_coeff_b = mf.mo_coeff
        mo_occ_a, mo_occ_b = mf.mo_occ
        nao, nmo = mo_coeff_a.shape
        nocc_a = (mo_occ_a>0).sum()
        nocc_b = (mo_occ_b>0).sum()
        nvir_a = nmo - nocc_a
        nvir_b = nmo - nocc_b

        x_a = mo_coeff_a
        x_inv_a = numpy.einsum('li,ls->is', x_a, mf.get_ovlp())
        x_t_inv_a = x_inv_a.T
        dm_mo_a = reduce(numpy.dot, (x_inv_a, dm_ao[0], x_t_inv_a))
        dm_mo_ov_a = dm_mo_a[:nocc_a, nocc_a:].reshape(nocc_a,nvir_a).T

        x_b = mo_coeff_b
        x_inv_b = numpy.einsum('li,ls->is', x_b, mf.get_ovlp())
        x_t_inv_b = x_inv_b.T
        dm_mo_b = reduce(numpy.dot, (x_inv_b, dm_ao[1], x_t_inv_b))
        dm_mo_ov_b = dm_mo_b[:nocc_b, nocc_b:].reshape(nocc_b,nvir_b).T

        
        xmy_a = [(tdscf.xy[i][0][0]-tdscf.xy[i][1][0]).reshape(nocc_a,nvir_a).T for i in range(len(tdscf.xy))]
        xmy_b = [(tdscf.xy[i][0][1]-tdscf.xy[i][1][1]).reshape(nocc_b,nvir_b).T for i in range(len(tdscf.xy))]
        
        xpy_a = [(tdscf.xy[i][0][0]+tdscf.xy[i][1][0]).reshape(nocc_a,nvir_a).T for i in range(len(tdscf.xy))]
        xpy_b = [(tdscf.xy[i][0][1]+tdscf.xy[i][1][1]).reshape(nocc_b,nvir_b).T for i in range(len(tdscf.xy))]

        dm_mo_ov_a = numpy.einsum("mjk,m,mi->ikj", xpy_a, am, numpy.cos(wmt))
        dm_mo_vo_a = numpy.einsum("mjk,m,mi->ijk", xpy_a, am, numpy.cos(wmt))

        dm_mo_ov_b = numpy.einsum("mjk,m,mi->ikj", xpy_b, am, numpy.cos(wmt))
        dm_mo_vo_b = numpy.einsum("mjk,m,mi->ijk", xpy_b, am, numpy.cos(wmt))

        dm_list = numpy.array([[dm_mo_a, dm_mo_b] for _ in t_array])
        dm_list[:, 0, :nocc_a,nocc_a:] = dm_mo_ov_a
        dm_list[:, 0, nocc_a:,:nocc_a] = dm_mo_vo_a
        dm_list[:, 1, :nocc_b,nocc_b:] = dm_mo_ov_b
        dm_list[:, 1, nocc_b:,:nocc_b] = dm_mo_vo_b


        return [[reduce(numpy.dot, (x_a, dm[0], x_a.T)), reduce(numpy.dot, (x_b, dm[1], x_b.T))] for dm in dm_list]
    
    else:
        mf = scf.addons.convert_to_rhf(mf)
        assert dm_ao.ndim == 2
        mo_coeff = mf.mo_coeff
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
        xpy = [(tdscf.xy[i][0]+tdscf.xy[i][1]).reshape(nocc,nvir).T for i in range(len(tdscf.xy))]

        dm_mo_ov = numpy.einsum("mjk,m,mi->ikj", xpy, am, numpy.cos(wmt))
        dm_mo_vo = numpy.einsum("mjk,m,mi->ijk", xpy, am, numpy.cos(wmt))

        dm_list = numpy.array([dm_mo for _ in t_array])
        dm_list[:, :nocc,nocc:] = dm_mo_ov
        dm_list[:, nocc:,:nocc] = dm_mo_vo

        return [reduce(numpy.dot, (x, dm, x.T)) for dm in dm_list]


if __name__ == "__main__":
    from pyscf import gto, scf, dft, tddft

    print("*******RKS*******")
    mol = gto.Mole()
    mol.atom = '''
    C         0.6584188440    0.0000000000    0.0000000000
    C        -0.6584188440    0.0000000000    0.0000000000
    H         1.2256624142    0.9143693597    0.0000000000
    H         1.2256624142   -0.9143693597    0.0000000000
    H        -1.2256624142    0.9143693597    0.0000000000
    H        -1.2256624142   -0.9143693597    0.0000000000'''
    mol.basis = 'sto-3g'
    mol.build()

    t_array = numpy.array([0.0, 0.1, 0.2])

    mf1 = dft.RKS(mol)
    mf1.xc = "PBE"
    mf1.max_cycle = 100
    mf1.conv_tol = 1e-12
    mf1.verbose = 0
    mf1.kernel()
    dm1 = mf1.make_rdm1()
    
    mf2 = dft.RKS(mol)
    mf2.xc = "pbe"
    mf2.max_cycle = 200
    mf2.conv_tol = 1e-12
    mf2.verbose = 0
    h1e_0 = mf2.get_hcore()
    e  = -1e-3
    ee = [e, 0, 0]
    ao_dip = mf2.mol.intor_symmetric('int1e_r', comp=3)
    ef = numpy.einsum('x,xij->ij', ee, ao_dip )
    h1e = h1e_0 + ef
    mf2.get_hcore = lambda *args: h1e
    mf2.kernel(dm0=dm1)
    dm2  = mf2.make_rdm1()

    td = tddft.TDDFT(mf1)
    td.nstates = 5
    td.max_cycle = 200
    td.verbose = 0
    td.kernel()
    print("td.e = ", td.e)
    am = proj_ex_states(td, dm2)
    print("am = \n", am)
    dms_rks = eval_rt_dm(td, dm2, am, td.e, t_array)
    print(dms_rks[0]-dm2)

    print("*******UKS*******")

    mf1 = dft.UKS(mol)
    mf1.xc = "PBE"
    mf1.max_cycle = 100
    mf1.conv_tol = 1e-12
    mf1.verbose = 0
    mf1.kernel()
    dm1 = mf1.make_rdm1()
    
    mf2 = dft.UKS(mol)
    mf2.xc = "pbe"
    mf2.max_cycle = 200
    mf2.conv_tol = 1e-12
    mf2.verbose = 0
    h1e_0 = mf2.get_hcore()
    e  = -1e-3
    ee = [e, 0, 0]
    ao_dip = mf2.mol.intor_symmetric('int1e_r', comp=3)
    ef = numpy.einsum('x,xij->ij', ee, ao_dip )
    h1e = h1e_0 + ef
    mf2.get_hcore = lambda *args: h1e
    mf2.kernel(dm0=dm1)
    dm2  = mf2.make_rdm1()

    td = tddft.TDDFT(mf1)
    td.nstates = 5
    td.max_cycle = 200
    td.verbose = 0
    td.kernel()
    am = proj_ex_states(td, dm2)
    print("td.e = ", td.e)
    print("am = \n", am)
    dms_uks = eval_rt_dm(td, dm2, am, td.e, t_array)

