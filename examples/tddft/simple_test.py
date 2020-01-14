#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example to run TDDFT calculation.
'''

import numpy
from pyscf import gto, scf, dft, tddft

mol = gto.Mole()
mol.build(
    atom = '''
  O    0.0000000    0.0184041   -0.0000000
  H   -0.0000000   -0.5383518    0.7830364
  H    0.0000000   -0.5383518   -0.7830364
  ''',  # in Angstrom
    basis = '3-21g',
    symmetry = True,
)

mf = scf.RHF(mol)
mf.kernel()

mytd = tddft.TDDFT(mf)
mytd.nstates = 5
mytd.kernel()
mytd.analyze()

print("\n\n")
a, b = mytd.get_ab()
print("a.shape = \n", a.shape)
print("b.shape = \n", b.shape)

mo_coeff = mf.mo_coeff
mo_energy = mf.mo_energy

mo_occ = mf.mo_occ
nao, nmo = mo_coeff.shape
occidx = numpy.where(mo_occ==2)[0]
viridx = numpy.where(mo_occ==0)[0]
nocc = len(occidx)
nvir = len(viridx)
nov  = nocc*nvir
print("nocc = %d, nvir = %d, nov  = %d"%(nocc, nvir, nov))
amat = a.reshape(nov, nov)
bmat = b.reshape(nov, nov)
abba = numpy.vstack(
    (
    numpy.hstack((amat, bmat)),
    numpy.hstack((-bmat, -amat))
    )
    )
print("eigvals = ", numpy.linalg.eigvals(abba) )
print("check if the eigenvals are in abba matrix, ", mytd.e)


orbv = mo_coeff[:,viridx]
orbo = mo_coeff[:,occidx]

xx = numpy.array(mytd.xy)[:,0,:,:]
yy = numpy.array(mytd.xy)[:,1,:,:]
print("xx.shape = ", xx.shape)
print("yy.shape = ", yy.shape)
print("mytd.e = ", mytd.e)

print("xt A x + xt B y + yt Bt x + yt At y = \n",
2*(
    numpy.einsum("kjb,iajb,lia->kl", xx, a, xx) + 
    numpy.einsum("kjb,iajb,lia->kl", xx, b, yy) + 
    numpy.einsum("kjb,iajb,lia->kl", yy, b, xx) + 
    numpy.einsum("kjb,iajb,lia->kl", yy, a, yy) )
)

print("xt x - yt y = \n",
2*(
    numpy.einsum("kia,lia->kl", xx, xx) + 
    # numpy.einsum("kjb,lia->kl", xx, -yy) + 
    # numpy.einsum("kjb,lia->kl", yy, xx) + 
    numpy.einsum("kia,lia->kl", yy, -yy) )
)

print("xt x = \n",
    2*numpy.einsum("kia,lia->kl", xx, xx)
)

print("yt y = \n",
    2*numpy.einsum("kia,lia->kl", xx, xx)
)

print("xt x + yt y = \n",
2*(
    numpy.einsum("kia,lia->kl", xx, xx) + 
    # numpy.einsum("kjb,lia->kl", xx, -yy) + 
    # numpy.einsum("kjb,lia->kl", yy, xx) + 
    numpy.einsum("kia,lia->kl", yy, yy) )
)
