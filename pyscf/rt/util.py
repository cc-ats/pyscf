import sys

import numpy
import scipy
from scipy.fftpack import fft
from pyscf import __config__

write = sys.stdout.write

DAMP_EXPO         = getattr(__config__, 'rt_tdscf_damp_expo',     1000)
PRINT_MAT_NCOL    = getattr(__config__, 'rt_tdscf_print_mat_ncol',   7)
DAMP_EXPO         = getattr(__config__, 'rt_tdscf_damp_expo',     1000)

def build_absorption_spectrum(tdscf, ndipole=None, damp_expo=DAMP_EXPO):
    if ndipole is None:
        ndipole = tdscf.ndipole

    ndipole_x = ndipole[:,0]
    ndipole_y = ndipole[:,1]
    ndipole_z = ndipole[:,2]
    
    ndipole_x = ndipole_x-ndipole_x[0]
    ndipole_y = ndipole_y-ndipole_y[0]
    ndipole_z = ndipole_z-ndipole_z[0]
    
    mw = 2.0 * numpy.pi * numpy.fft.fftfreq(
        tdscf.ntime.size, tdscf.dt
    )
    damp = numpy.exp(-tdscf.ntime/damp_expo)
    fwx = numpy.fft.fft(ndipole_x*damp)
    fwy = numpy.fft.fft(ndipole_y*damp)
    fwz = numpy.fft.fft(ndipole_z*damp)
    fw = (fwx.imag + fwy.imag + fwz.imag) / 3.0 
    sigma = - mw * fw
    mm = mw.size
    m  = mm//2

    mw = mw[:m]
    sigma = sigma[:m]
    scale = numpy.abs(sigma.max())
    return mw, sigma/scale

def print_matrix(title, array_, ncols=7, fmt='% 11.6e'):
    ''' printing a real rectangular matrix, or the real part of a complex matrix, ncols columns per batch '''
    array = array_.real
    write(title+'\n')
    m = array.shape[1]
    n = array.shape[0]
    #write('m=%d n=%d\n' % (m, n))
    nbatches = int(n/ncols)
    if nbatches * ncols < n: nbatches += 1
    for k in range(0, nbatches):
        write('     ')  
        j1 = ncols*k
        j2 = ncols*(k+1)
        if k == nbatches-1: j2 = n 
        for j in range(j1, j2):
            write('   %6d  ' % (j+1))
        write('\n')
        for i in range(0, m): 
            write(' %2d -' % (i+1))
            for j in range(j1, j2):
                write( fmt % array[j,i])
            write('\n')

def print_cx_matrix(title, cx_array_, ncols=7, fmt=' % 11.6e'):
    ''' printing a complex rectangular matrix, ncols columns per batch '''
    print_matrix(title+" Real Part ", cx_array_.real, ncols=ncols, fmt=fmt)
    print_matrix(title+" Imag Part ", cx_array_.imag, ncols=ncols, fmt=fmt)

def errm(m1,m2):
    ''' check consistency '''
    n   = numpy.linalg.norm(m1-m2)
    r   = m1.size
    e   = n/r
    return numpy.abs(e)

def expm(m, do_bch=False):
    if not do_bch:
        return scipy.linalg.expm(m)
    else:
        raise NotImplementedError("BCH not implemented here")

if __name__ == "__main__":
    from pyscf import gto, scf
    mol =   gto.Mole( atom='''
    O    0.0000000    0.0000000    0.5754646
    O    0.0000000    0.0000000   -0.5754646
    '''
    , basis='cc-pvdz', spin=2, symmetry=False).build()

    mf = scf.UHF(mol)
    mf.verbose = 5
    mf.kernel()

    dm = mf.make_rdm1()
    print_matrix("dm = ", dm[0])
