import time
import numpy
import pyscf
from pyscf import lib
from pyscf import gto
from pyscf import df
from pyscf import dft
from pyscf.dft import numint, xcfun
from pyscf import __config__

def calc_rho(mf, coords, dm, ao_value=None):
    mol = mf.mol
    if ao_value is None:
        ao_value = numint.eval_ao(mol, coords, deriv=0)
        return numint.eval_rho(mol, ao_value, dm, xctype='LDA')
    else:
        if len(ao_value.shape) == 2:
            return numint.eval_rho(mol, ao_value, dm, xctype='LDA')
        else:
            return numint.eval_rho(mol, ao_value[0], dm, xctype='LDA')

def calc_rho_t(mf, coords, dm, ao_value=None):
    mol = mf.mol
    if ao_value is None:
        ao_value = numint.eval_ao(mol, coords, deriv=2)
    if len(ao_value.shape) == 3 and ao_value.shape[0]==10:
        rho = numint.eval_rho(mol, ao_value, dm, xctype='mGGA')
        rhot = rho[5]  
        return rhot
    else:
        raise RuntimeError("Wrong AO value")

def calc_rhov_rhoj(mf, coords, dm):
    mol = mf.mol
    rhov = 0
    for i in range(mol.natm):
        r = mol.atom_coord(i)
        Z = mol.atom_charge(i)
        rp = r - coords
        rhov += Z / numpy.einsum('xi,xi->x', rp, rp)**.5

    rhoj = numpy.empty_like(rhov)
    for p0, p1 in lib.prange(0, rhoj.size, 600):
        fakemol = gto.fakemol_for_charges(coords[p0:p1])
        ints = df.incore.aux_e2(mol, fakemol)
        rhoj[p0:p1] = lib.einsum('ijp,ij->p', ints, dm)
    return rhov, - 0.5*rhoj

def calc_rhok(mf, coords, dm):
    nbas = mf.mol.nbas
    ngrids = coords.shape[0]
    rhok = numpy.zeros(ngrids)
    for ip0, ip1 in lib.prange(0, ngrids, 600):
        fakemol = gto.fakemol_for_charges(coords[ip0:ip1])
        pmol = mf.mol + fakemol
        ints = pmol.intor('int3c2e_sph', shls_slice=(0,nbas,0,nbas,nbas,nbas+fakemol.nbas))
        esp_ao = gto.intor_cross('int1e_ovlp', mf.mol, fakemol)
        esp = lib.einsum('mp,mu->up', esp_ao, dm)
        rhok[ip0:ip1] = -numpy.einsum('mp,up,mup->p', esp, esp, ints)
    return rhok

def calc_rhok_lr(mf, coords, dm):
    ks = mf
    ni = ks._numint
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(ks.xc, spin=mol.spin)
    nbas = ks.mol.nbas
    ngrids = coords.shape[0]
    rhok_lr = numpy.zeros(ngrids)
    with ks.mol.with_range_coulomb(omega):
        for ip0, ip1 in lib.prange(0, ngrids, 600):
            fakemol = gto.fakemol_for_charges(coords[ip0:ip1])
            pmol = ks.mol + fakemol
            ints = pmol.intor('int3c2e_sph', shls_slice=(0,nbas,0,nbas,nbas,nbas+fakemol.nbas))
            esp_ao = gto.intor_cross('int1e_ovlp', ks.mol, fakemol)
            esp = lib.einsum('mp,mu->up', esp_ao, dm)
            rhok_lr[ip0:ip1] = -numpy.einsum('mp,up,mup->p', esp, esp, ints)
    return rhok_lr

def calc_rhoxc(mf, coords, dm, ao_value=None):
    mol = mf.mol
    if ao_value is None:
        ao_value = numint.eval_ao(mol, coords, deriv=2)
    rho = numint.eval_rho(mol, ao_value, dm, xctype='mGGA')
    if hasattr(mf, 'xc'):
        ks = mf
        ni = ks._numint
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(ks.xc, spin=mol.spin)
        rhoxc = dft.libxc.eval_xc(mf.xc, rho)[0]
        if abs(hyb) < 1e-10 and abs(alpha) < 1e-10:
            return rhoxc*rho[0]
        else:
            rhok = calc_rhok(mf, coords, dm)
            if abs(omega) > 1e-10:
                rhok_lr = calc_rhok_lr(mf, coords, dm)
                return rhoxc*rho[0] + 0.25*hyb*rhok + 0.25*(alpha-hyb)*rhok_lr
            else:
                return rhoxc*rho[0] + 0.25*hyb*rhok
    else:
        return 0.25*calc_rhok(mf, coords, dm)


def calc_rho_ene(mf, coords, dm, ao_value=None):
    mol = mf.mol
    if ao_value is None:
        ao_value = numint.eval_ao(mol, coords, deriv=2)
    if len(ao_value.shape) == 3 and ao_value.shape[0]==10:
        rho = numint.eval_rho(mol, ao_value, dm, xctype='mGGA')
        rhot = rho[5]
        rhoxc = calc_rhoxc(mf, coords, dm, ao_value=ao_value)
        rhov, rhoj = calc_rhov_rhoj(mf, coords, dm)
        return (rhoxc + rhot - (rhov + rhoj)*rho[0])
    else:
        raise RuntimeError("Wrong AO value")

if __name__ == '__main__':
    from pyscf import gto, scf
    from pyscf.dft import gen_grid
    from pyscf.tools import cubegen

    mol = gto.M(atom='''O 0.00000000,  0.000000,  0.000000
                H 0.761561, 0.478993, 0.00000000
                H -0.761561, 0.478993, 0.00000000'''
                , basis='6-31g*')

    grids = gen_grid.Grids(mol)
    grids.build()
    coords  = grids.coords
    weights = grids.weights

    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    dm = mf.make_rdm1()

    ao_value = numint.eval_ao(mol, coords, deriv=2)
    rho = calc_rho(mf, coords, dm, ao_value=ao_value)
    print("N elec = %f, ref N elec = 10"
    %(lib.einsum("i,i->", weights, rho))
    )

    print("HF")
    rhoe = calc_rho_ene(mf, coords, dm, ao_value=ao_value)
    print("E elec = %f, ref E elec = %f"
    %(lib.einsum("i,i->", weights, rhoe), mf.energy_elec()[0]))

    mf = scf.RKS(mol)
    mf.xc = 'BLYP'
    mf.verbose = 0
    mf.kernel()
    dm = mf.make_rdm1()

    rhoe = calc_rho_ene(mf, coords, dm, ao_value=ao_value)
    print('BLYP')
    print("E elec = %f, ref E elec = %f"
    %(lib.einsum("i,i->", weights, rhoe), mf.energy_elec()[0]))

    mf.xc = 'b3lyp'
    mf.kernel()
    dm = mf.make_rdm1()

    rhoe = calc_rho_ene(mf, coords, dm, ao_value=ao_value)
    print('b3lyp')
    print("E elec = %f, ref E elec = %f"
    %(lib.einsum("i,i->", weights, rhoe), mf.energy_elec()[0]))

    mf.xc = 'camb3lyp'
    mf.kernel()
    dm = mf.make_rdm1()

    rhoe = calc_rho_ene(mf, coords, dm, ao_value=ao_value)
    print('cam-b3lyp')
    print("E elec = %f, ref E elec = %f"
    %(lib.einsum("i,i->", weights, rhoe), mf.energy_elec()[0]))
