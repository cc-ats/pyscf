/* Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
  
   Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
 
        http://www.apache.org/licenses/LICENSE-2.0
 
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

 *
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
//#include <omp.h>
#include "config.h"
#include "cint.h"
#include "optimizer.h"
#include "nr_direct.h"

int GTOmax_shell_dim(const int *ao_loc, const int *shls_slice, int ncenter);
int GTOmax_cache_size(int (*intor)(), int *shls_slice, int ncenter,
                      int *atm, int natm, int *bas, int nbas, double *env);

#define AO_BLOCK_SIZE   32

#define MIN(I,J)        ((I) < (J) ? (I) : (J))

#define DECLARE_ALL \
        const int *atm = envs->atm; \
        const int *bas = envs->bas; \
        const double *env = envs->env; \
        const int natm = envs->natm; \
        const int nbas = envs->nbas; \
        const int *ao_loc = envs->ao_loc; \
        const int *shls_slice = envs->shls_slice; \
        const CINTOpt *cintopt = envs->cintopt; \
        const int ioff = ao_loc[shls_slice[0]]; \
        const int joff = ao_loc[shls_slice[2]]; \
        const int koff = ao_loc[shls_slice[4]]; \
        const int loff = ao_loc[shls_slice[6]]; \
        const int ish0 = ishls[0]; \
        const int ish1 = ishls[1]; \
        const int jsh0 = jshls[0]; \
        const int jsh1 = jshls[1]; \
        const int ksh0 = kshls[0]; \
        const int ksh1 = kshls[1]; \
        const int lsh0 = lshls[0]; \
        const int lsh1 = lshls[1]; \
        int shls[4]; \
        void (*pf)(double *eri, double *dm, JKArray *vjk, int *shls, \
                   int i0, int i1, int j0, int j1, \
                   int k0, int k1, int l0, int l1); \
        int (*fprescreen)(); \
        if (vhfopt) { \
                fprescreen = vhfopt->fprescreen; \
        } else { \
                fprescreen = CVHFnoscreen; \
        } \
        int ish, jsh, ksh, lsh, i0, j0, k0, l0, i1, j1, k1, l1, idm;

#define INTOR_AND_CONTRACT \
        shls[0] = ish; \
        shls[1] = jsh; \
        shls[2] = ksh; \
        shls[3] = lsh; \
        if ((*fprescreen)(shls, vhfopt, atm, bas, env) \
            && (*intor)(buf, NULL, shls, atm, natm, bas, nbas, env, \
                        cintopt, cache)) { \
                i0 = ao_loc[ish] - ioff; \
                j0 = ao_loc[jsh] - joff; \
                k0 = ao_loc[ksh] - koff; \
                l0 = ao_loc[lsh] - loff; \
                i1 = ao_loc[ish+1] - ioff; \
                j1 = ao_loc[jsh+1] - joff; \
                k1 = ao_loc[ksh+1] - koff; \
                l1 = ao_loc[lsh+1] - loff; \
                for (idm = 0; idm < n_dm; idm++) { \
                        pf = jkop[idm]->contract; \
                        (*pf)(buf, dms[idm], vjk[idm], shls, \
                              i0, i1, j0, j1, k0, k1, l0, l1); \
                } \
        }

/*
 * for given ksh, lsh, loop all ish, jsh
 */
void CVHFdot_nrs1(int (*intor)(), JKOperator **jkop, JKArray **vjk,
                  double **dms, double *buf, double *cache, int n_dm,
                  int *ishls, int *jshls, int *kshls, int *lshls,
                  CVHFOpt *vhfopt, IntorEnvs *envs)
{
        DECLARE_ALL;

        for (ish = ish0; ish < ish1; ish++) {
        for (jsh = jsh0; jsh < jsh1; jsh++) {
        for (ksh = ksh0; ksh < ksh1; ksh++) {
        for (lsh = lsh0; lsh < lsh1; lsh++) {
                INTOR_AND_CONTRACT;
        } } } }
}

void CVHFdot_nrs2ij(int (*intor)(), JKOperator **jkop, JKArray **vjk,
                    double **dms, double *buf, double *cache, int n_dm,
                    int *ishls, int *jshls, int *kshls, int *lshls,
                    CVHFOpt *vhfopt, IntorEnvs *envs)
{
        if (ishls[0] > jshls[0]) {
                return CVHFdot_nrs1(intor, jkop, vjk, dms, buf, cache, n_dm,
                                    ishls, jshls, kshls, lshls, vhfopt, envs);
        } else if (ishls[0] == jshls[0]) {

                DECLARE_ALL;

                for (ish = ish0; ish < ish1; ish++) {
                for (jsh = jsh0; jsh <= ish; jsh++) {
                for (ksh = ksh0; ksh < ksh1; ksh++) {
                for (lsh = lsh0; lsh < lsh1; lsh++) {
                        INTOR_AND_CONTRACT;
                } } } }
        }
}

void CVHFdot_nrs2kl(int (*intor)(), JKOperator **jkop, JKArray **vjk,
                    double **dms, double *buf, double *cache, int n_dm,
                    int *ishls, int *jshls, int *kshls, int *lshls,
                    CVHFOpt *vhfopt, IntorEnvs *envs)
{
        if (kshls[0] > lshls[0]) {
                return CVHFdot_nrs1(intor, jkop, vjk, dms, buf, cache, n_dm,
                                    ishls, jshls, kshls, lshls, vhfopt, envs);
        } else if (kshls[0] == lshls[0]) {
                assert(kshls[1] == lshls[1]);

                DECLARE_ALL;

                for (ish = ish0; ish < ish1; ish++) {
                for (jsh = jsh0; jsh < jsh1; jsh++) {
                for (ksh = ksh0; ksh < ksh1; ksh++) {
                for (lsh = lsh0; lsh <= ksh; lsh++) {
                        INTOR_AND_CONTRACT;
                } } } }
        }
}

void CVHFdot_nrs4(int (*intor)(), JKOperator **jkop, JKArray **vjk,
                  double **dms, double *buf, double *cache, int n_dm,
                  int *ishls, int *jshls, int *kshls, int *lshls,
                  CVHFOpt *vhfopt, IntorEnvs *envs)
{
        if (ishls[0] > jshls[0]) {
                return CVHFdot_nrs2kl(intor, jkop, vjk, dms, buf, cache, n_dm,
                                      ishls, jshls, kshls, lshls, vhfopt, envs);
        } else if (ishls[1] <= jshls[0]) {
                return;
        } else if (kshls[0] > lshls[0]) {  // ishls == jshls
                return CVHFdot_nrs2ij(intor, jkop, vjk, dms, buf, cache, n_dm,
                                      ishls, jshls, kshls, lshls, vhfopt, envs);
        } else if (kshls[0] == lshls[0]) {  // ishls == jshls
                assert(kshls[1] == lshls[1]);

                DECLARE_ALL;

                for (ish = ish0; ish < ish1; ish++) {
                for (jsh = jsh0; jsh <= ish; jsh++) {
                for (ksh = ksh0; ksh < ksh1; ksh++) {
                for (lsh = lsh0; lsh <= ksh; lsh++) {
                        INTOR_AND_CONTRACT;
                } } } }
        }
}

void CVHFdot_nrs8(int (*intor)(), JKOperator **jkop, JKArray **vjk,
                  double **dms, double *buf, double *cache, int n_dm,
                  int *ishls, int *jshls, int *kshls, int *lshls,
                  CVHFOpt *vhfopt, IntorEnvs *envs)
{
        if (ishls[0] > kshls[0]) {
                return CVHFdot_nrs4(intor, jkop, vjk, dms, buf, cache, n_dm,
                                    ishls, jshls, kshls, lshls, vhfopt, envs);
        } else if (ishls[0] < kshls[0]) {
                assert(ishls[1] == kshls[1]);
                return;
        } else if ((ishls[1] <= jshls[0]) || (kshls[1] <= lshls[0])) {
                return;
        }
        // else i == k && i >= j && k >= l

        DECLARE_ALL;

        for (ish = ish0; ish < ish1; ish++) {
        for (jsh = jsh0; jsh < MIN(jsh1, ish+1); jsh++) {
        for (ksh = ksh0; ksh <= ish; ksh++) {
        for (lsh = lsh0; lsh < MIN(lsh1, ksh+1); lsh++) {
/* when ksh==ish, (lsh<jsh) misses some integrals (eg k<i&&l>j).
 * These integrals are calculated in the next (ish,jsh) pair. To show
 * that, we just need to prove that every elements in shell^4 appeared
 * only once in fjk_s8.  */
                if ((ksh == ish) && (lsh > jsh)) {
                        break;
                }
                INTOR_AND_CONTRACT;
        } } } }
}

static JKArray *allocate_JKArray(JKOperator *op, int *shls_slice, int *ao_loc, int ncomp)
{
        JKArray *jkarray = malloc(sizeof(JKArray));
        int ibra = op->ibra_shl0;
        int iket = op->iket_shl0;
        int obra = op->obra_shl0;
        int oket = op->oket_shl0;
        int v_bra_sh0 = shls_slice[obra];
        int v_ket_sh0 = shls_slice[oket];
        int v_bra_sh1 = shls_slice[obra+1];
        int v_ket_sh1 = shls_slice[oket+1];
        jkarray->v_ket_nsh  = shls_slice[oket+1] - shls_slice[oket];
        jkarray->dm_dims[0] = ao_loc[shls_slice[ibra+1]] - ao_loc[shls_slice[ibra]];
        jkarray->dm_dims[1] = ao_loc[shls_slice[iket+1]] - ao_loc[shls_slice[iket]];
        int v_rows = ao_loc[v_bra_sh1] - ao_loc[v_bra_sh0];
        int v_cols = ao_loc[v_ket_sh1] - ao_loc[v_ket_sh0];
        jkarray->offset0_outptr = v_bra_sh0 * jkarray->v_ket_nsh + v_ket_sh0;
        int outptr_size =((shls_slice[obra+1] - shls_slice[obra]) *
                          (shls_slice[oket+1] - shls_slice[oket]));
        jkarray->outptr = malloc(sizeof(int) * outptr_size);
        memset(jkarray->outptr, NOVALUE, sizeof(int) * outptr_size);
        jkarray->stack_size = 0;
        int data_size = v_rows * v_cols * ncomp;
        jkarray->data = malloc(sizeof(double) * data_size);
        jkarray->ncomp = ncomp;
        return jkarray;
}

static void deallocate_JKArray(JKArray *jkarray)
{
        free(jkarray->outptr);
        free(jkarray->data);
        free(jkarray);
}

static double *allocate_and_reorder_dm(JKOperator *op, double *dm,
                                       int *shls_slice, int *ao_loc)
{
        int ibra = op->ibra_shl0;
        int iket = op->iket_shl0;
        int ish0 = shls_slice[ibra];
        int jsh0 = shls_slice[iket];
        int ish1 = shls_slice[ibra+1];
        int jsh1 = shls_slice[iket+1];
        int ioff = ao_loc[ish0];
        int joff = ao_loc[jsh0];
        int nrow = ao_loc[ish1] - ioff;
        int ncol = ao_loc[jsh1] - joff;
        double *out = malloc(sizeof(double) * nrow*ncol);
        int ish, jsh, i0, i1, j0, j1, i, j, ij;

        ij = 0; 
        for (ish = ish0; ish < ish1; ish++) {
        for (jsh = jsh0; jsh < jsh1; jsh++) {
                i0 = ao_loc[ish  ] - ioff;
                i1 = ao_loc[ish+1] - ioff;
                j0 = ao_loc[jsh  ] - joff;
                j1 = ao_loc[jsh+1] - joff;
                for (i = i0; i < i1; i++) {
                for (j = j0; j < j1; j++, ij++) {
                        out[ij] = dm[i*ncol+j];
                } }
        } }
        return out;
}

static void zero_out_vjk(double *vjk, JKOperator *op,
                         int *shls_slice, int *ao_loc, int ncomp)
{
        int obra = op->obra_shl0;
        int oket = op->oket_shl0;
        int ish0 = shls_slice[obra];
        int jsh0 = shls_slice[oket];
        int ish1 = shls_slice[obra+1];
        int jsh1 = shls_slice[oket+1];
        int nbra = ao_loc[ish1] - ao_loc[ish0];
        int nket = ao_loc[jsh1] - ao_loc[jsh0];
        memset(vjk, 0, sizeof(double) * nbra * nket * ncomp);
}

static void assemble_v(double *vjk, JKOperator *op, JKArray *jkarray,
                       int *shls_slice, int *ao_loc)
{
        int obra = op->obra_shl0;
        int oket = op->oket_shl0;
        int ish0 = shls_slice[obra];
        int jsh0 = shls_slice[oket];
        int ish1 = shls_slice[obra+1];
        int jsh1 = shls_slice[oket+1];
        int njsh = jsh1 - jsh0;
        size_t vrow = ao_loc[ish1] - ao_loc[ish0];
        size_t vcol = ao_loc[jsh1] - ao_loc[jsh0];
        int ncomp = jkarray->ncomp;
        int voffset = ao_loc[ish0] * vcol + ao_loc[jsh0];
        int i, j, ish, jsh;
        int di, dj, icomp;
        int optr;
        double *data, *pv;

        for (ish = ish0; ish < ish1; ish++) {
        for (jsh = jsh0; jsh < jsh1; jsh++) {
                optr = jkarray->outptr[ish*njsh+jsh-jkarray->offset0_outptr];
                if (optr != NOVALUE) {
                        di = ao_loc[ish+1] - ao_loc[ish];
                        dj = ao_loc[jsh+1] - ao_loc[jsh];
                        data = jkarray->data + optr;
                        pv = vjk + ao_loc[ish]*vcol+ao_loc[jsh] - voffset;
                        for (icomp = 0; icomp < ncomp; icomp++) {
                                for (i = 0; i < di; i++) {
                                for (j = 0; j < dj; j++) {
                                        pv[i*vcol+j] += data[i*dj+j];
                                } }
                                pv += vrow * vcol;
                                data += di * dj;
                        }
                }
        } }
}

// Divide shls into subblocks with roughly equal number of AOs in each block
static int shls_block_partition(int *block_loc, int *shls_slice, int *ao_loc)
{
        int ish0 = shls_slice[0];
        int ish1 = shls_slice[1];
        int ao_loc_last = ao_loc[ish0];
        int count = 1;
        int ish;

        block_loc[0] = ish0;
        for (ish = ish0 + 1; ish < ish1; ish++) {
                if (ao_loc[ish] - ao_loc_last > AO_BLOCK_SIZE) {
                        block_loc[count] = ish;
                        count++;
                        ao_loc_last = ao_loc[ish];
                }
        }
        block_loc[count] = ish1;
        return count;
}



/*
 * drv loop over ij, generate eris of kl for given ij, call fjk to
 * calculate vj, vk.
 * 
 * n_dm is the number of dms for one [array(ij|kl)], it is also the size of dms and vjk
 * ncomp is the number of components that produced by intor
 * shls_slice = [ishstart, ishend, jshstart, jshend, kshstart, kshend, lshstart, lshend]
 *
 * ao_loc[i+1] = ao_loc[i] + CINTcgto_spheric(i, bas)  for i = 0..nbas
 *
 * Return [(ptr[ncomp,nao,nao] in C-contiguous) for ptr in vjk]
 */
void CVHFnr_direct_drv(int (*intor)(), void (*fdot)(), JKOperator **jkop,
                       double **dms, double **vjk, int n_dm, int ncomp,
                       int *shls_slice, int *ao_loc,
                       CINTOpt *cintopt, CVHFOpt *vhfopt,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        IntorEnvs envs = {natm, nbas, atm, bas, env, shls_slice, ao_loc, NULL,
                cintopt, ncomp};
        int idm;
        double *tile_dms[n_dm];
        for (idm = 0; idm < n_dm; idm++) {
                zero_out_vjk(vjk[idm], jkop[idm], shls_slice, ao_loc, ncomp);
                tile_dms[idm] = allocate_and_reorder_dm(jkop[idm], dms[idm],
                                                        shls_slice, ao_loc);
        }

        const int di = GTOmax_shell_dim(ao_loc, shls_slice, 4);
        const int cache_size = GTOmax_cache_size(intor, shls_slice, 4,
                                                 atm, natm, bas, nbas, env);
        int *block_iloc = malloc(sizeof(int) * (shls_slice[1] + shls_slice[3] +
                                                shls_slice[5] + shls_slice[7] + 4));
        int *block_jloc = block_iloc + shls_slice[1] + 1;
        int *block_kloc = block_jloc + shls_slice[3] + 1;
        int *block_lloc = block_kloc + shls_slice[5] + 1;
        const int nblock_i = shls_block_partition(block_iloc, shls_slice+0, ao_loc);
        const int nblock_j = shls_block_partition(block_jloc, shls_slice+2, ao_loc);
        const int nblock_k = shls_block_partition(block_kloc, shls_slice+4, ao_loc);
        const int nblock_l = shls_block_partition(block_lloc, shls_slice+6, ao_loc);
        const int nblock_kl = nblock_k * nblock_l;
        const int nblock_jkl = nblock_j * nblock_kl;
        const int nblock_ijkl = nblock_i * nblock_jkl;

#pragma omp parallel
{
        int i, j, k, l, r, blk_id;
        JKArray *v_priv[n_dm];
        for (i = 0; i < n_dm; i++) {
                v_priv[i] = allocate_JKArray(jkop[i], shls_slice, ao_loc, ncomp);
        }
        double *buf = malloc(sizeof(double) * (di*di*di*di*ncomp + cache_size));
        double *cache = buf + di*di*di*di*ncomp;
#pragma omp for nowait schedule(dynamic, 1)
        for (blk_id = 0; blk_id < nblock_ijkl; blk_id++) {
                // dispatch blk_id to sub-block indices (i, j, k, l)
                r = blk_id;
                i = r / nblock_jkl; r = r - i * nblock_jkl;
                j = r / nblock_kl ; r = r - j * nblock_kl;
                k = r / nblock_l  ; r = r - k * nblock_l;
                l = r;
                (*fdot)(intor, jkop, v_priv, tile_dms, buf, cache, n_dm,
                        block_iloc+i, block_jloc+j, block_kloc+k, block_lloc+l,
                        vhfopt, &envs);
        }
#pragma omp critical
        {
                for (i = 0; i < n_dm; i++) {
                        assemble_v(vjk[i], jkop[i], v_priv[i], shls_slice, ao_loc);
                        deallocate_JKArray(v_priv[i]);
                }
        }
        free(buf);
}
        for (idm = 0; idm < n_dm; idm++) {
                free(tile_dms[idm]);
        }
        free(block_iloc);
}

