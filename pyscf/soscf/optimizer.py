import numpy
from numpy     import dot
from functools import reduce

class OptimizeAlgorithm(object):
    pass

class LineSearch(OptimizeAlgorithm):
    def __init__(self):
        self.scale = 0.1

        self._alpha_left  = None
        self._g_left      = None
        self._y_left      = None
        self._alpha_right = None
        self._g_right     = None
        self._y_right     = None

        self.is_first_step = True
        self.is_line_search_step_fail = False

    def reset(self):
        self._alpha_left  = None
        self._g_left      = None
        self._y_left      = None
        self._alpha_right = None
        self._g_right     = None
        self._y_right     = None
        self.is_first_step = True

    def next_step(self, y_prev, alpha_prev, g_prev, 
                        y_next, alpha_next, g_next,
                        is_wolfe1, is_wolfe2):
        if self.is_first_step:
            self._alpha_left   = alpha_prev
            self._g_left       = g_prev
            self._y_left       = y_prev
            self._alpha_right  = None
            self._g_right      = None
            self._y_right      = None
            self.is_first_step = False

            assert alpha_next >= 0.0 and g_prev < 0.0
            assert not (is_wolfe1 and is_wolfe2)
            self.is_line_search_step_fail = False

        if not is_wolfe1:
            alpha_diff = alpha_next - self._alpha_left
            alpha_new  = -self._g_left * alpha_diff * alpha_diff 
            alpha_new /= 2.0*(y_next-self._y_left-self._g_left*alpha_diff)

            self._alpha_right = alpha_next
            self._g_right     = g_next
            self._y_right     = y_next
            alpha_len = self._alpha_right - self._alpha_left

            if self._alpha_right is not None:
                if alpha_new > self._alpha_right:
                    self.is_line_search_step_fail = True
                elif alpha_new < self._alpha_left + self.scale * alpha_len:
                    alpha_new = self._alpha_left + self.scale * alpha_len
                elif alpha_new > self._alpha_right - self.scale * alpha_len:
                    alpha_new = self._alpha_right - self.scale * alpha_len

        elif not is_wolfe2:
            alpha_new  = alpha_next * self._g_left 
            alpha_new += self._alpha_left * g_next
            alpha_new /= self._g_left - g_next
            alpha_len = alpha_next - self._alpha_left

            if alpha_new < self._alpha_left or alpha_new > self._alpha_left + 10.0 * alpha_len:
                alpha_new = self._alpha_left + 10.0 * alpha_len
            elif alpha_new < self._alpha_left + 3.0 * alpha_len:
                alpha_new = self._alpha_left + 3.0 * alpha_len
            
            self._alpha_left = alpha_next
            self._g_left     = g_next
            self._y_left     = y_next

            if self._alpha_right is not None:
                if alpha_new > (1 - self.scale) * self._alpha_right + self.scale * self._alpha_left:
                    alpha_new = (1 - self.scale) * self._alpha_right + self.scale * self._alpha_left

        assert not self.is_line_search_step_fail
        return alpha_new

class DogLegSearch(OptimizeAlgorithm):
    def __init__(self):
        self._r_trust        = None
        self._delta_y_model  = None
        
        self.default_r_trust = 0.33
        self.scale_down      = 0.33
        self.scale_up        = 2.0
        self.tol_low         = 0.8
        self.tol_high        = 1.2

        self.do_update_r_trust = True
        self.is_first_step     = True

        self.step_info = None

    def reset(self):
        self._r_trust          = None
        self._delta_y_model    = None
        self.is_first_step     = True

    def get_trust_radius(self, delta_y):
        if self.is_first_step:
            self.is_first_step = False
            r_trust       = self.default_r_trust
            self._r_trust = r_trust
            return r_trust
        else:
            assert self._delta_y_model is not None
            delta_y_model = self._delta_y_model

            if not self.do_update_r_trust:
                r_trust = self._r_trust
                return r_trust
            else:
                if delta_y > 0.0:
                    r_trust = self._r_trust * self.scale_down
                elif (delta_y/delta_y_model > self.tol_low) and (delta_y/delta_y_model < self.tol_high):
                    r_trust = self._r_trust * self.scale_up
                    r_trust = min(r_trust, self.default_r_trust)
                else:
                    r_trust = self._r_trust
                self._r_trust = r_trust
                return r_trust
    
    def next_step(self, subspace):
        self.step_info = None
        if self._r_trust is not None:
            r_trust = self._r_trust
        else:
            r_trust = self.default_r_trust
        # BFGS build hess_inv
        dim_subspace, tmp_num = subspace.shape
        num_subspace_vec      = (tmp_num+1)//2
        gmat = subspace[:,:num_subspace_vec]
        smat = subspace[:,num_subspace_vec:(2*num_subspace_vec)]

        hess_inv = numpy.eye(dim_subspace)
        for i_subspace_vec in range(1,num_subspace_vec):
            g_new = gmat[:,i_subspace_vec]
            g_old = gmat[:,i_subspace_vec-1]
            
            yk    = (g_new - g_old).reshape(dim_subspace,1)
            sk    = (smat[:,i_subspace_vec-1]).reshape(dim_subspace,1)
            
            yk_dot_sk = dot(yk.T,sk)
            sk_dot_sk = dot(sk.T,sk)
            yk_dot_yk = dot(yk.T,yk)
            inv_hess_thresh = numpy.sqrt(sk_dot_sk) * numpy.sqrt(yk_dot_yk)
            
            if numpy.abs(yk_dot_sk) >= 1e-8 * inv_hess_thresh:
                u = numpy.eye(dim_subspace) - dot(sk, yk.T)/yk_dot_sk
                hess_inv = reduce(dot, [u, hess_inv, u.T]) + dot(sk,sk.T)/yk_dot_sk
            
        eig_vals, eig_vecs = numpy.linalg.eigh(hess_inv)
        abs_eig_vals       = numpy.abs(eig_vals)
        hess_inv           = reduce(dot, [eig_vecs, numpy.diag(abs_eig_vals), eig_vecs.T])

        dim_subspace, tmp_num = subspace.shape
        num_subspace_vec      = (tmp_num+1)//2
        
        sd_step      = subspace[:,num_subspace_vec-1].reshape(dim_subspace,1)
        
        qn_step      = dot(hess_inv, sd_step)
        
        norm_sd_step = numpy.linalg.norm(sd_step)
        norm_qn_step = numpy.linalg.norm(qn_step)

        if norm_qn_step < r_trust:
            self.step_info = "DogLegSearch: Normal BFGS step"
            step_vec = sd_step
        elif norm_sd_step > r_trust:
            self.step_info  = "DogLegSearch: Descent step"
            sd_step        *= r_trust/norm_sd_step
            step_vec = sd_step
        else:
            self.step_info  = "DogLegSearch: BFGS step"
            dl_step = qn_step - sd_step
            dl_dot_dl = numpy.dot(dl_step.T, dl_step)
            sd_dot_dl = numpy.dot(sd_step.T, dl_step)

            a = dl_dot_dl
            b = 2.0 * sd_dot_dl
            sd2 = norm_sd_step * norm_sd_step
            r2  = r_trust * r_trust
            c = sd2 - r2
            d = b*b - 4*a*c
            assert d >= 0
            x = (numpy.sqrt(d)-b)/(2.0*a)
            step_vec = sd_step + x * dl_step
        try:
            sol_vec = numpy.linalg.solve(hess_inv,step_vec)
            
            self._delta_y_model  = -numpy.dot(step_vec.T, sd_step)[0]
            self._delta_y_model -= 0.5 * numpy.dot(step_vec.T, sol_vec)[0]
        except numpy.linalg.LinAlgError:
            sol_vec = numpy.zeros_like(step_vec)
            self._delta_y_model = -numpy.dot(step_vec.T, sd_step)[0]
        return step_vec.reshape(-1)