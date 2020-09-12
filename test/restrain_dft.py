import numpy as np
import scipy.linalg as la
import copy
from functools import reduce
from pyscf import gto, scf, lo, dft, lib

def get_localized_orbitals(mf, lo_method, mo=None):
    if mo is None:
        mo = mf.mo_coeff

    mol = mf.mol
    s1e = mf.get_ovlp()

    if lo_method.lower() == 'lowdin' or lo_method.lower() == 'meta_lowdin':
        C = lo.orth_ao(mf, 'meta_lowdin', s=s1e)
        C_inv = np.dot(C.conj().T,s1e)
        if isinstance(mf, scf.hf.RHF):
            C_inv_spin = C_inv
        else:
            C_inv_spin = np.array([C_inv]*2)

    elif lo_method == 'iao':
        s1e = mf.get_ovlp()
        pmol = mf.mol.copy()
        pmol.build(False, False, basis='minao')
        if isinstance(mf, scf.hf.RHF):
            mo_coeff_occ = mf.mo_coeff[:,mf.mo_occ>0]
            C = lo.iao.iao(mf.mol, mo_coeff_occ)
            # Orthogonalize IAO
            C = lo.vec_lowdin(C, s1e)
            C_inv = np.dot(C.conj().T,s1e)
            C_inv_spin = C_inv
        else:
            mo_coeff_occ_a = mf.mo_coeff[0][:,mf.mo_occ[0]>0]
            mo_coeff_occ_b = mf.mo_coeff[1][:,mf.mo_occ[1]>0]
            C_a = lo.iao.iao(mf.mol, mo_coeff_occ_a)
            C_b = lo.iao.iao(mf.mol, mo_coeff_occ_b)
            C_a = lo.vec_lowdin(C_a, s1e)
            C_b = lo.vec_lowdin(C_b, s1e)
            C_inv_a = np.dot(C_a.T, s1e)
            C_inv_b = np.dot(C_b.T, s1e)
            C_inv_spin = np.array([C_inv_a, C_inv_b])

    elif lo_method == 'nao':
        C = lo.orth_ao(mf, 'nao')
        C_inv = np.dot(C.conj().T,s1e)
        if isinstance(mf, scf.hf.RHF):
            C_inv_spin = C_inv
        else:
            C_inv_spin = np.array([C_inv]*2)

    else:
        raise NotImplementedError("UNDEFINED LOCAL ORBITAL TYPE, EXIT...")

    mo_lo = np.einsum('...jk,...kl->...jl', C_inv_spin, mo)
    return C_inv_spin, mo_lo

def pop_analysis(mf, mo_on_loc_ao, disp=True, full_dm=False):
    '''
    population analysis for local orbitals.
    return dm_lo

    mf should be a converged object
    full_rdm = False: return the diagonal element of dm_lo
    disp = True: show all the population to screen
    '''
    dm_lo = mf.make_rdm1(mo_on_loc_ao, mf.mo_occ)

    if disp:
        mf.mulliken_pop(mf.mol, dm_lo, np.eye(mf.mol.nao_nr()))

    if full_dm:
        return dm_lo
    else:
        return np.einsum('...ii->...i', dm_lo)

def get_fock_add_rdft(restraints, v, c_ao2lo_inv):
    v_lagr           = constraints.sum2separated(v)
    sites_a, sites_b = constraints.unique_sites()
    if isinstance(mf, scf.hf.RHF):
        c_ao2lo_a = c_ao2lo_b = c_ao2lo_inv
    else:
        c_ao2lo_a, c_ao2lo_b = c_ao2lo_inv

    v_a = np.einsum('ip,i,iq->pq', c_ao2lo_a[sites_a].conj(), v_lagr[0], c_ao2lo_a[sites_a])
    v_b = np.einsum('ip,i,iq->pq', c_ao2lo_b[sites_b].conj(), v_lagr[1], c_ao2lo_b[sites_b])

    if isinstance(mf, scf.hf.RHF):
        return v_a + v_b
    else:
        return np.array((v_a, v_b))

def v_rdft(mf, restraints, orb_pop):
    if isinstance(mf, scf.hf.RHF):
        pop_a = pop_b = orb_pop * .5
    else:
        pop_a, pop_b  = orb_pop
    
    n_c              = restraints.nelec_required
    omega_vals       = restraints.omega_vals
    sites_a, sites_b = restraints.unique_sites()

    n_cur     = pop_a[sites_a], pop_b[sites_b]
    n_cur_sum = restraints.separated2sum(n_cur)[1]

    return omega_vals*(n_cur_sum - n_c)

def rdft(mf, restraints, lo_method='lowdin', tol=1e-5,
         constraints_tol=1e-3, maxiter=200, C_inv=None, verbose=4,
         diis_pos='post', diis_type=1):

    mf.verbose   = verbose
    mf.max_cycle = maxiter

    if not mf.converged:
        mf.kernel()

    old_get_fock = mf.get_fock
    c_inv        = get_localized_orbitals(mf, lo_method, mf.mo_coeff)[0]

    cdft_diis       = lib.diis.DIIS()
    cdft_diis.space = 8

    if lo_method.lower() == 'iao':
        mo_on_loc_ao = get_localized_orbitals(mf, lo_method, mf.mo_coeff)[1]
    else:
        mo_on_loc_ao = np.einsum('...jk,...kl->...jl', c_inv, mf.mo_coeff)

    orb_pop = pop_analysis(mf, mo_on_loc_ao, disp=False)
    v_0     = v_rdft(mf, restraints, orb_pop)

    def get_fock(h1e, s1e, vhf, dm, cycle=0, mf_diis=None):
        fock_0 = old_get_fock(h1e, s1e, vhf, dm, cycle, None)

        if mf_diis is None:
            fock_add = get_fock_add_rdft(restraints, v_0, c_inv)
            return fock_0 + fock_add

        fock_0   = old_get_fock(h1e, s1e, vhf, dm, cycle, None)
        fock_add = get_fock_add_rdft(restraints, v_0, c_inv)
        fock     = fock_0 + fock_add 

        return fock

    dm0 = mf.make_rdm1()
    mf.get_fock = get_fock
    mf.kernel(dm0)

    mo_on_loc_ao = get_localized_orbitals(mf, lo_method, mf.mo_coeff)[1]
    orb_pop = pop_analysis(mf, mo_on_loc_ao, disp=True)
    return mf, orb_pop

class PopulationScheme(object):
    

class Restraints(object):
    def __init__(self, orbital_indices, spin_labels, nelec_required, omega_vals):
        self.orbital_indices = orbital_indices
        self.spin_labels     = spin_labels
        self.nelec_required  = np.asarray(nelec_required)
        self.omega_vals     = np.asarray(omega_vals)
        assert(
        len(orbital_indices) == len(spin_labels) == len(nelec_required) == len(omega_vals)
            )

    def get_n_constraints(self):
        return len(self.nelec_required)

    def unique_sites(self):
        sites_a = []
        sites_b = []
        for group, spin_labels in zip(self.orbital_indices, self.spin_labels):
            for orbidx, spin in zip(group, spin_labels):
                if spin == 0:
                    sites_a.append(orbidx)
                else:
                    sites_b.append(orbidx)
        sites_a = np.sort(list(set(sites_a)))
        sites_b = np.sort(list(set(sites_b)))
        return sites_a, sites_b

    def site_to_constraints_transform_matrix(self):
        sites_a, sites_b = self.unique_sites()
        map_sites_a = dict(((v,k) for k,v in enumerate(sites_a)))
        map_sites_b = dict(((v,k) for k,v in enumerate(sites_b)))

        n_constraints = self.get_n_constraints()
        t_a = np.zeros((sites_a.size, n_constraints))
        t_b = np.zeros((sites_b.size, n_constraints))
        for k, group in enumerate(self.orbital_indices):
            for orbidx, spin in zip(group, self.spin_labels[k]):
                if spin == 0:
                    t_a[map_sites_a[orbidx],k] += 1
                else:
                    t_b[map_sites_b[orbidx],k] += 1
        return t_a, t_b

    def sum2separated(self, pop_vals):
        '''
        convert the format of constraint from a summation format (it allows several orbitals' linear combination)
        to the format each orbital is treated individually (also they are separated by spin)
        '''
        t_a, t_b = self.site_to_constraints_transform_matrix()
        v_c = self.omega_vals * pop_vals
        v_c_a = np.einsum('pi,i->p', t_a, v_c)
        v_c_b = np.einsum('pi,i->p', t_b, v_c)
        return v_c_a, v_c_b

    def separated2sum(self, n_c):
        '''the inversion function for sum2separated'''
        t_a, t_b = self.site_to_constraints_transform_matrix()
        n_c_new = np.array([np.einsum('pi,p->i', t_a, n_c[0]),
                            np.einsum('pi,p->i', t_b, n_c[1])]).T

        n_c_sum = n_c_new[:,0] + n_c_new[:,1]
        v_c_sum = [n_c_new[i,0] if 0 in spins else n_c_new[i,1]
                   for i,spins in enumerate(self.spin_labels)]
        return n_c_new, n_c_sum, v_c_sum

if __name__ == '__main__':
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = '''
    He 0.0 0.0 -2.0
    H  0.0 0.0 -2.929352
    He 0.0 0.0 2.0
    H  0.0 0.0 2.929352
    '''
    mol.basis = 'sto-3g'
    mol.spin   = 0
    mol.charge = 2
    mol.build()

    mf           = scf.RHF(mol)
    mf.conv_tol  = 1e-9
    mf.verbose   = 0
    mf.max_cycle = 100
    mf.kernel()

    orbital_indices = [[0,0],[2,2]]
    spin_labels     = [[0,1],[0,1]]
    nelec_required  = [1,1]
    omega_vals      = [9.9, 0.1]
    constraints = Restraints(orbital_indices, spin_labels, nelec_required, omega_vals)
    mf, dm_pop  = rdft(mf, constraints, lo_method='lowdin', verbose=4)
