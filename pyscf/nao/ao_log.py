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

from __future__ import print_function, division
import numpy as np
import copy
from scipy.spatial.distance import cdist

from pyscf.nao.log_mesh import log_mesh

#
#
#
class ao_log(log_mesh):
    '''
    Holder of radial orbitals on logarithmic grid.
    Args:
      ions : list of ion structures (read from ion files from siesta)
        or 
      gto  : gaussian type of orbitals object from pySCF
        or
      gpaw : ??

    Returns:
      ao_log:
        sp2ion (ion structure from m_siesta_ion or m_siesta_ion_xml):
          List of structure composed of several field read from the ions file.
        nr (int): number of radial point
        rmin (float)
        kmax (float)
        rmax (float)
        rr
        pp
        psi_log
        psi_log_rl
        sp2rcut (array, float): array containing the rcutoff of each specie
        sp_mu2rcut (array, float)
        interp_rr instance of log_interp_c to interpolate along real-space axis
        interp_pp instance of log_interp_c to interpolate along momentum-space axis

    Examples:
    '''
    def __init__(self, **kw):
      """ Initializes numerical orbitals  """

      log_mesh.__init__(self, **kw)
      if 'ao_log' in kw : # this is creating a deepcopy of all attributes
        ao = kw['ao_log']
        for a in ao.__dict__.keys():
          try:    setattr(self, a, copy.deepcopy(getattr(ao, a)))
          except: pass

      elif 'gto' in kw:
        self.init_ao_log_gto(**kw)
      elif 'sp2ion' in kw:
        self.init_ao_log_ion(**kw)
      elif 'setups' in kw:
        self.init_ao_log_gpaw(**kw)
      elif 'xyz_list' in kw:
        pass
      else:
        print(kw.keys())
        raise RuntimeError('unknown initialization method')
      
      
    def init_ao_log_gto(self, **kw):
      """ supposed to be private """
      import numpy as np
      from pyscf.nao.m_log_interp import log_interp_c

      self.interp_rr,self.interp_pp = log_interp_c(self.rr), log_interp_c(self.pp)

      gto,rcut_tol = kw['gto'],self.rcut_tol
      a2s = [gto.atom_symbol(ia) for ia in range(gto.natm) ]
      self.sp2symbol = sorted(list(set(a2s)))
      nspecies = self.nspecies = len(self.sp2symbol)
      atom2sp = np.array([self.sp2symbol.index(s) for s in a2s], dtype=np.int64)
      self.sp2charge = np.array([-999]*self.nspecies, dtype=np.int64)
      for ia,sp in enumerate(atom2sp): self.sp2charge[sp]=gto.atom_charge(ia)

      self.sp_mu2j = [0]*nspecies
      self.psi_log = [0]*nspecies
      self.psi_log_rl = [0]*nspecies
      self.sp2nmult = np.zeros(nspecies, dtype=np.int64)
     
      seen_species = [] # this is auxiliary to organize the loop over species 
      for ia,sp in enumerate(atom2sp):
        if sp in seen_species: continue
        seen_species.append(sp)
        self.sp2nmult[sp] = nmu = sum([gto.bas_nctr(sid) for sid in gto.atom_shell_ids(ia)])

        mu2ff = np.zeros((nmu, self.nr))
        mu2ff_rl = np.zeros((nmu, self.nr))
        mu2j = np.zeros(nmu, dtype=np.int64)
        mu = -1
        for sid in gto.atom_shell_ids(ia):
          pows, coeffss = gto.bas_exp(sid), gto.bas_ctr_coeff(sid)
          for coeffs in coeffss.T:
            mu=mu+1
            l = mu2j[mu] = gto.bas_angular(sid)
            for ir, r in enumerate(self.rr):
              mu2ff_rl[mu,ir] = sum(pows[:]**((2*l+3)/4.0)*coeffs[:]*np.exp(-pows[:]*r**2))
              mu2ff[mu,ir] = r**l*mu2ff_rl[mu,ir]
              
        self.sp_mu2j[sp] = mu2j
        norms = [np.sqrt(self.interp_rr.dg_jt*sum(ff**2*self.rr**3)) for ff in mu2ff]
        for mu,norm in enumerate(norms): 
          mu2ff[mu,:] = mu2ff[mu,:]/norm
          mu2ff_rl[mu,:] = mu2ff_rl[mu,:]/norm

        self.psi_log[sp] = mu2ff
        self.psi_log_rl[sp] = mu2ff_rl
      
      self.jmx = max([mu2j.max() for mu2j in self.sp_mu2j])
      
      self.sp_mu2s = []
      for mu2j in self.sp_mu2j:
        mu2s = np.zeros(len(mu2j)+1, dtype=np.int64)
        for mu,j in enumerate(mu2j): mu2s[mu+1] = mu2s[mu]+2*j+1
        self.sp_mu2s.append(mu2s)
      
      self.sp2norbs = np.array([mu2s[-1] for mu2s in self.sp_mu2s])
      self.sp_mu2rcut = []
      for sp, mu2ff in enumerate(self.psi_log):
        mu2rcut = np.zeros(len(mu2ff))
        for mu,ff in enumerate(mu2ff):
          ffmx,irmx = abs(mu2ff[mu]).max(), abs(mu2ff[mu]).argmax()
          irrp = np.argmax(abs(ff[irmx:])<ffmx*rcut_tol)
          irrc = irmx+irrp if irrp>0 else -1
          mu2rcut[mu] = self.rr[irrc]
        self.sp_mu2rcut.append(mu2rcut)
      self.sp2rcut = np.array([mu2rcut.max() for mu2rcut in self.sp_mu2rcut])
      return self

    #
    #  
    def init_ao_log_ion(self, **kw):
        """
            Reads data from a previous SIESTA calculation,
            interpolates the Pseudo-Atomic Orbitals on a single log mesh.
        """
        from pyscf.nao.m_log_interp import log_interp_c
        from pyscf.nao.m_spline_diff2 import spline_diff2
        from pyscf.nao.m_spline_interp import spline_interp

        self.interp_rr,self.interp_pp = log_interp_c(self.rr), log_interp_c(self.pp)
        sp2ion = self.sp2ion = kw['sp2ion']
        fname = kw['fname'] if 'fname' in kw else 'paos'
      
        if fname in sp2ion[0]: 
            self.siesta_ion_interp(sp2ion, fname=fname)
    
        self.sp2vna = [None]*len(sp2ion)     # Interpolate a Neutral-Atom potential V_NA(r) for each specie
        self.sp2rcut_vna = np.zeros(len(sp2ion))
        for isp, ion in enumerate(sp2ion):
            
            if "vna" not in ion.keys(): continue
            if ion["vna"] is None: continue

            self.sp2rcut_vna[isp] = ion["vna"]["cutoff"]
            h,dat = ion["vna"]["delta"][0], ion["vna"]["data"][0][:,1]
            d2 = spline_diff2(h, dat, 0.0, 1.0e301)
            self.sp2vna[isp] = np.array([0.5*spline_interp(h, dat, d2, r) for r in self.rr]) # given in Rydberg in sp2ion

        self.sp2chlocal = [None]*len(sp2ion)     # Interpolate the atomic charges for each specie 
        self.sp2rcut_chlocal = np.zeros(len(sp2ion))
        for isp, ion in enumerate(sp2ion):
            
            if "chlocal" not in ion.keys(): continue
            if ion["chlocal"] is None: continue

            self.sp2rcut_chlocal[isp] = ion["chlocal"]["cutoff"]
            h,dat = ion["chlocal"]["delta"][0], ion["chlocal"]["data"][0][:,1]
            d2 = spline_diff2(h, dat, 0.0, 1.0e301)
            self.sp2chlocal[isp] = np.array([spline_interp(h, dat, d2, r) for r in self.rr]) 

        return self

    #
    #
    #
    def siesta_ion_interp(self, sp2ion, fname='paos'):
        from pyscf.nao.m_get_sp_mu2s import get_sp_mu2s
        from pyscf.nao.m_spline_diff2 import spline_diff2
        from pyscf.nao.m_spline_interp import spline_interp
        """
        Interpolation of orbitals or projectors given on linear grid in the ion dictionary  
            rr : is the grid on which we want the function
            sp2ion : list of dictionaries
            fname : function name, can be 'paos' or 'kbs'
        """
        rr, nr, nsp = self.rr, len(self.rr), len(sp2ion)
        pname = {'paos': 'orbital', 'kbs': 'projector'}[fname]

        self.nspecies = len(sp2ion)
        self.sp2nmult = np.zeros(self.nspecies, dtype='int64')
        self.sp_mu2rcut = [None]*self.nspecies
        self.sp_mu2j = [None]*self.nspecies
        self.sp_mu2s = [None]*self.nspecies
        self.sp2norbs = np.zeros(self.nspecies, dtype='int64')
        self.sp2rcut = np.zeros(self.nspecies)
        self.sp2charge = np.zeros(self.nspecies, dtype='int64')
        self.sp2valence = np.zeros(self.nspecies, dtype='int64')
        
        for isp, ion in enumerate(sp2ion):
            if ion[fname] is None:
                continue
            self.sp2nmult[isp] = len(ion[fname]['data'])
            self.sp_mu2rcut[isp] = np.array(ion[fname]["cutoff"])
            self.sp_mu2j[isp] = np.array([o["l"] for o in ion[fname][pname]], dtype='int64')
            
            mu2s = np.zeros(self.sp2nmult[isp]+1, dtype='int64')
            for mu in range(self.sp2nmult[isp]):
                mu2s[mu+1] = sum(2*self.sp_mu2j[isp][0:mu+1]+1)
            self.sp_mu2s[isp] = mu2s
            self.sp2norbs[isp] = self.sp_mu2s[isp][self.sp2nmult[isp]]
            self.sp2rcut[isp] = np.amax(self.sp_mu2rcut[isp])
            self.sp2charge[isp] = int(self.sp2ion[isp]['z'])
            self.sp2valence[isp] = int(self.sp2ion[isp]['valence'])

        self.jmx = max([mu2j.max() for mu2j in self.sp_mu2j if mu2j is not None])

        if fname=='kbs':
            self.sp_mu2vkb = [None]*self.nspecies
            for isp, ion in enumerate(sp2ion):
                if ion[fname] is None:
                    continue
                self.sp_mu2vkb[isp] = np.array([0.5*p['ref_energy'] for p in ion['kbs']['projector'] ])
    
        self.psi_log_rl = [None]*self.nspecies
        for isp, ion in enumerate(sp2ion):
            if ion[fname] is None:
                continue
            ff = np.zeros((len(ion[fname][pname]), nr))
            for mu,(h,dat) in enumerate(zip(ion[fname]["delta"],ion[fname]["data"])):
                diff2 = spline_diff2(h, dat[:,1], 0.0, 1.0e301)
                for i, r in enumerate(rr):
                    ff[mu,i] = spline_interp(h, dat[:,1], diff2, r)
            self.psi_log_rl[isp] = ff

        self.psi_log = [None]*self.nspecies
        for isp, (mu2ff, mu2j) in enumerate(zip(self.psi_log_rl, self.sp_mu2j)):
            if mu2ff is None:
                continue
            gg = np.zeros((len(mu2j), nr))
            for mu,(ff,j) in enumerate(zip(mu2ff,mu2j)): 
                gg[mu] = ff*(rr**j)
            self.psi_log[isp] = gg


    def init_ao_log_gpaw(self, **kw):
      """ Reads radial orbitals from a previous GPAW calculation. """
      from pyscf.nao.m_log_interp import log_interp_c

      #self.setups = setups if setup is saved in ao_log, we get the following error
      #                           while performing copy
      #    File "/home/marc/anaconda2/lib/python2.7/copy.py", line 182, in deepcopy
      #    rv = reductor(2)
      #    TypeError: can't pickle Spline objects

      self.interp_rr,self.interp_pp = log_interp_c(self.rr), log_interp_c(self.pp)
      setups = kw['setups']
      sdic = setups.setups
      self.sp2key = sdic.keys()
      #key0 = sdic.keys()[0]
      #print(key0, sdic[key0].Z, dir(sdic[key0]))
      self.sp_mu2j = [np.array(sdic[key].l_orb_j, np.int64) for key in sdic.keys()]
      self.sp2nmult = np.array([len(mu2j) for mu2j in self.sp_mu2j], dtype=np.int64)
      self.sp2charge = np.array([sdic[key].Z for key in sdic.keys()], dtype=np.int64)
      self.nspecies = len(self.sp_mu2j)
      self.jmx = max([max(mu2j) for mu2j in self.sp_mu2j])
      self.sp2norbs = np.array([sum(2*mu2j+1) for mu2j in self.sp_mu2j], dtype=np.int64)
      self.sp_mu2rcut = []
      self.psi_log_rl = []
      self.psi_log = []


      for sp,[key,nmu,mu2j] in enumerate(zip(sdic.keys(), self.sp2nmult, self.sp_mu2j)):
        self.sp_mu2rcut.append(np.array([phit.get_cutoff() for phit in sdic[key].phit_j]))
        mu2ff = np.zeros([nmu, self.nr])
        for mu,phit in enumerate(sdic[key].phit_j):
          for ir, r in enumerate(self.rr):
              mu2ff[mu,ir],deriv = phit.get_value_and_derivative(r)

        self.psi_log_rl.append(mu2ff)
        self.psi_log.append(mu2ff* (self.rr**mu2j[mu]))
      
      self.sp2rcut = np.array([np.amax(rcuts) for rcuts in self.sp_mu2rcut], dtype='float64') # derived from sp_mu2rcut

      self.sp_mu2s = []  # derived from sp_mu2j
      for mu2j in self.sp_mu2j:
        mu2s = np.zeros(len(mu2j)+1, dtype=np.int64)
        for mu,j in enumerate(mu2j): mu2s[mu+1] = mu2s[mu]+2*j+1
        self.sp_mu2s.append(mu2s)

      #self._add_sp2info()
      #self._add_psi_log_mom()
      
      #print(self.sp_mu2j)
      #print(self.sp2nmult)
      #print(self.nspecies)
      #print(self.jmx)
      #print(self.sp2norbs)
      #print(self.sp_mu2rcut)
      #print(self.psi_log)
      #print(self.sp2charge)
      #print(self.sp2rcut)
      #print(self.sp_mu2s)

      return self

    #
    def _add_sp2info(self):
      """ Adds a field sp2info containing, for each specie lists of integer charcteristics: """
      self.sp2info = []
      for sp,[mu2j,mu2s] in enumerate(zip(self.sp_mu2j,self.sp_mu2s)):
        self.sp2info.append([ [mu, j, mu2s[mu], mu2s[mu+1]] for mu,j in enumerate(mu2j)])

    #
    def _add_psi_log_mom(self):
      """ Adds a field psi_log_mom which contains Bessel transforms of original radial functions (from psi_log) """

      import numpy as np
      from pyscf.nao.m_sbt import sbt_c
      
      sbt = sbt_c(self.rr, self.pp, lmax=self.jmx)
      self.psi_log_mom = []

      for sp,[nmu,mu2ff,mu2j] in enumerate(zip(self.sp2nmult, self.psi_log, self.sp_mu2j)):
        mu2ao = np.zeros((nmu,self.nr), dtype='float64')
        for mu,[am,ff] in enumerate(zip(mu2j,mu2ff)): mu2ao[mu,:] = sbt.sbt( ff, am, 1 )
        self.psi_log_mom.append(mu2ao)
      del sbt
    
    # 
    def view(self):
      """ Shows a plot of all radial orbitals """
      import matplotlib.pyplot as plt
      for sp in range(self.nspecies):
        plt.figure(sp+1)
        plt.title('Orbitals for specie='+ str(sp)+' Znuc='+str(self.sp2charge[sp]))
        for j,ff in zip(self.sp_mu2j[sp], self.psi_log[sp]):
          if j>0 :
            plt.plot(self.rr, ff, '--', label=str(j))
          else:
            plt.plot(self.rr, ff, '-', label=str(j))
        #plt.xlim([0.0,3.0])
        plt.legend()
      
      plt.show()

    def comp_moments(self):
      """
        Computes the scalar and dipole moments of the product functions
        Args:
          argument can be  prod_log    or   ao_log
      """
      rr3dr = self.rr**3*np.log(self.rr[1]/self.rr[0])
      rr4dr = self.rr*rr3dr
      sp2mom0,sp2mom1,cs,cd = [],[],np.sqrt(4*np.pi),np.sqrt(4*np.pi/3.0)
      for sp,nmu in enumerate(self.sp2nmult):
        nfunct=sum(2*self.sp_mu2j[sp]+1)
        mom0 = np.zeros((nfunct))
        d = np.zeros((nfunct,3))
        for mu,[j,s] in enumerate(zip(self.sp_mu2j[sp],self.sp_mu2s[sp])):
          if j==0:                 mom0[s]  = cs*sum(self.psi_log[sp][mu,:]*rr3dr)
          if j==1: d[s,1]=d[s+1,2]=d[s+2,0] = cd*sum(self.psi_log[sp][mu,:]*rr4dr)
        sp2mom0.append(mom0)
        sp2mom1.append(d)
      return sp2mom0,sp2mom1
    
    def get_aoneo(self):
      """Packs the data into one array for a later transfer to the library """
      import numpy as np
      from numpy import require, float64, concatenate as conc
      nr  = self.nr
      nsp = self.nspecies
      nmt = sum(self.sp2nmult)    
      nrt = nr*nmt
      nms = nmt+nsp

      nsvn = 200 + 2*nr + 4*nsp + 2*nmt + nrt + nms
      svn = require(np.zeros(nsvn), dtype=float64, requirements='CW')
      # Simple parameters
      i = 0
      svn[i] = nsp;        i+=1;
      svn[i] = nr;         i+=1;
      svn[i] = self.rmin;  i+=1;
      svn[i] = self.rmax;  i+=1;
      svn[i] = self.kmax;  i+=1;
      svn[i] = self.jmx;   i+=1;
      svn[i] = conc(self.psi_log).sum(); i+=1;
      # Pointers to data
      i = 99
      s = 199
      svn[i] = s+1; i+=1; f=s+nr;  svn[s:f] = self.rr; s=f; # pointer to rr
      svn[i] = s+1; i+=1; f=s+nr;  svn[s:f] = self.pp; s=f; # pointer to pp
      svn[i] = s+1; i+=1; f=s+nsp; svn[s:f] = self.sp2nmult; s=f; # pointer to sp2nmult
      svn[i] = s+1; i+=1; f=s+nsp; svn[s:f] = self.sp2rcut;  s=f; # pointer to sp2rcut
      svn[i] = s+1; i+=1; f=s+nsp; svn[s:f] = self.sp2norbs; s=f; # pointer to sp2norbs
      svn[i] = s+1; i+=1; f=s+nsp; svn[s:f] = self.sp2charge; s=f; # pointer to sp2charge    
      svn[i] = s+1; i+=1; f=s+nmt; svn[s:f] = conc(self.sp_mu2j); s=f; # pointer to sp_mu2j
      svn[i] = s+1; i+=1; f=s+nmt; svn[s:f] = conc(self.sp_mu2rcut); s=f; # pointer to sp_mu2rcut
      svn[i] = s+1; i+=1; f=s+nrt; svn[s:f] = conc(self.psi_log).reshape(nrt); s=f; # pointer to psi_log
      svn[i] = s+1; i+=1; f=s+nms; svn[s:f] = conc(self.sp_mu2s); s=f; # pointer to sp_mu2s
      svn[i] = s+1; # this is a terminator to simple operation
      return svn

    #
    #
    #
    def ao_eval(self, ra, isp, coords):
      from pyscf.nao.m_rsphar_libnao import rsphar
      """
        Compute the values of atomic orbitals on given grid points
        Args:
          ao  : instance of ao_log_c class
          ra  : vector where the atomic orbitals from "ao" are centered
          isp : specie index for which we compute
          coords: coordinates on which we compute
        Returns:
          res[norbs,ncoord] : array of atomic orbital values
      """
      rrs = cdist(ra.reshape((1,3)), coords).reshape(-1)
      rcutmx = self.sp2rcut[isp]
      mu_c2pao = self.interp_rr.interp_csr(self.psi_log_rl[isp], rrs, rcut=rcutmx)
      
      #print(__name__, mu_c2pao.shape)
      
      res = np.zeros((self.sp2norbs[isp],coords.shape[0]))
      jmx_sp = self.sp_mu2j[isp].max()
      rsh = np.zeros((jmx_sp+1)**2)
      for icrd,(coord,r) in enumerate(zip(coords-ra, rrs)):
        if rrs[icrd]>rcutmx: continue
        rsphar(coord, jmx_sp, rsh)
      
        for mu,(j,s,f) in enumerate(zip(self.sp_mu2j[isp],self.sp_mu2s[isp],self.sp_mu2s[isp][1:])):
          fval = mu_c2pao[mu,icrd] if j==0 else mu_c2pao[mu,icrd] * r**j
          res[s:f,icrd] = fval * rsh[j*(j+1)-j:j*(j+1)+j+1]
      return res

#
#
#
if __name__=="__main__":
  from pyscf import gto
  from pyscf.nao.m_ao_log import ao_log_c
  """ Interpreting small Gaussian calculation """
  mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz') # coordinates in Angstrom!
  ao = ao_log(gto=mol)
  print(ao.sp2norbs)
  
