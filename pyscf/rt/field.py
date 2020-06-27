# Author: Junjie Yang <yangjunjie0320@gmail.com> Zheng Pei and Yihan Shao
import numpy
from numpy import dot, zeros, array, asarray, complex128, einsum
from numpy import exp, cos, sin, power

from pyscf import gto, scf, lib

x_base_vec = array([1,0,0])
y_base_vec = array([0,1,0])
z_base_vec = array([0,1,0])

def zero_field_vec(t):
    return [0.0, 0.0, 0.0]

def gaussian_field_vec(t, freq, sigma, mu, strength_vec):
    amplitude = exp(-power(t - mu, 2) / (2 * power(sigma, 2))) * cos(freq*t)
    return amplitude*asarray(strength_vec)

def oscillation_field_vec(t, freq, strength_vec):
    amplitude = cos(freq*t)
    return amplitude*asarray(strength_vec)

class Field(lib.StreamObject):
    """
    Field base class.
    """
    pass

class ClassicalElectricField(Field):
    """
    A class which manages classical electric field perturbations.
    """
    def __init__(self, mol, field_func=zero_field_vec, stop_time=None):
        self.mol        = mol
        self.field_func = field_func
        self.stop_time  = stop_time
        self.dip_ints   = mol.intor("cint1e_r_sph", comp=3).astype(numpy.complex128)

    def get_field_vec(self, t, field_func=None):
        if (field_func is None):
            field_func =  self.field_func
        
        if (field_func is None):
            return zeros(3, dtype=complex128)
        elif (self.stop_time is not None):
            if t > self.stop_time:
                return zeros(3, dtype=complex128)
            else:
                vec = field_func(t)
                return asarray(vec, dtype=complex128).reshape(3)
        else:
            vec = field_func(t)
            return asarray(vec, dtype=complex128).reshape(3)

    def get_field_ao(self, t, field_vec=None, dip_ints=None):
        if field_vec is None:
            field_vec = self.get_field_vec(t)
        if dip_ints is None:
            dip_ints = self.dip_ints

        return einsum('x,xij->ij', field_vec, dip_ints)

if __name__ == "__main__":
    h2o_631g = gto.Mole() # water
    h2o_631g.atom = '''
    O   -0.0000000   -0.0514481   -0.0000000
    H   -0.0000000    0.5547343    0.7830366
    H    0.0000000    0.5547343   -0.7830366
    '''
    h2o_631g.basis = '6-31g'
    h2o_631g.build()
    dip_ints = h2o_631g.intor("cint1e_r_sph", comp=3).astype(numpy.complex128)

    gau_vec = lambda t: gaussian_field_vec(t, 1.0, 1.0, 0.0, [0.2,0.1,3.0])
    gaussian_field = ClassicalElectricField(h2o_631g, field_func=gau_vec, stop_time=10.0)

    ref_field_vec = gaussian_field_vec(1.0, 1.0, 1.0, 0.0, [0.2,0.1,3.0])
    assert numpy.linalg.norm(gaussian_field.get_field_vec(1.0) - ref_field_vec)==0
    ref_field_ao = dip_ints[0]*ref_field_vec[0] + dip_ints[1]*ref_field_vec[1] + dip_ints[2]*ref_field_vec[2]
    assert numpy.linalg.norm(gaussian_field.get_field_ao(1.0) - ref_field_ao)==0

    ref_field_vec = array([0.0, 0.0, 0.0])
    assert numpy.linalg.norm(gaussian_field.get_field_vec(12.0) - ref_field_vec)==0
    ref_field_ao = dip_ints[0]*ref_field_vec[0] + dip_ints[1]*ref_field_vec[1] + dip_ints[2]*ref_field_vec[2]
    assert numpy.linalg.norm(gaussian_field.get_field_ao(12.0) - ref_field_ao)==0