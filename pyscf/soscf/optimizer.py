import numpy
from numpy import dot

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

        assert alpha_next > 0.0 and g_prev < 0.0
        assert not (is_wolfe1 and is_wolfe2)
        is_line_search_step_fail = False

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
                    is_line_search_step_fail = True
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

            if alpha_new > (1 - self.scale) * self._alpha_right + self.scale * self._alpha_left:
                alpha_new = (1 - self.scale) * self._alpha_right + self.scale * self._alpha_left

        assert not is_line_search_step_fail
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
    
    def next_step(self, sd_step, qn_step, hess_inv):
        assert self._r_trust is not None
        r_trust = self._r_trust

        norm_sd_step = numpy.linalg.norm(sd_step)
        norm_qn_step = numpy.linalg.norm(qn_step)

        if norm_qn_step < r_trust:
            print(" Normal BFGS step")
            step_vec = sd_step
        elif norm_sd_step > r_trust:
            sd_step *= r_trust/norm_sd_step
            print(" Descent step")
            step_vec = sd_step
        else:
            print(" Dog-leg BFGS step")
            dl_step = qn_step - sd_step
            dl_dot_dl = numpy.dot(dl_step, dl_step)
            sd_dot_dl = numpy.dot(sd_step, dl_step)

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
            self._delta_y_model  = -numpy.dot(step_vec, sd_step)
            self._delta_y_model -= 0.5 * numpy.dot(step_vec, sol_vec)
        except numpy.linalg.LinAlgError:
            sol_vec = numpy.zeros_like(step_vec)
            self._delta_y_model = -numpy.dot(step_vec, sd_step)
        return step_vec



def main(f, gradf, hessf, x, alpha=5.0, beta=1e-4, sigma=0.1, max_line_search=15):
    line_search_obj   = LineSearch()
    dog_leg_obj       = DogLegSearch()

    step       = None
    alpha_cur  = None
    alpha_prev = None
    alpha_low  = None
    alpha_high = None
    
    x_prev = None
    x_cur  = x
    y_prev = None
    y_cur  = None

    grad_cur  = None
    grad_list = []
    step_cur  = None
    step_list = []

    g_prev = None
    g_cur  = None

    is_wolfe1 = False
    is_wolfe2 = False
    is_first  = True

    iter_line_search = 0
    nssv             = 0

    while True:
        y_cur    = f(x_cur)
        grad_cur = gradf(x_cur)
        hess_cur = hessf(x_cur)
        hinv_cur = numpy.linalg.inv(hess_cur)
        rms      = numpy.linalg.norm(grad_cur)/grad_cur.size
        if rms < beta * beta:
            return True

        if is_first:
            step_cur = numpy.empty_like(grad_cur)
        else:
            y_diff = y_cur - y_prev
            if -y_diff > beta*alpha*g_prev:
                is_wolfe1 = True
            else:
                is_wolfe1 = False
            g_cur     = numpy.dot(step_cur, grad_cur)
            alpha2    = numpy.dot(step_cur, step_cur)
            alpha_cur = numpy.sqrt(alpha2)
            g_cur     = g_cur/alpha_cur

            if g_cur >= 0.9*g_prev:
                is_wolfe2 = True
                if iter_line_search > max_line_search:
                    is_wolfe2 = True
                    is_first  = True
            else:
                is_wolfe2 = False
            
            if is_wolfe1 and is_wolfe2:
                iter_line_search = 0
                nssv += 1
            else:
                if not is_wolfe1:
                    print("Line search: overstep")
                else:
                    print("Line search: understep")
                iter_line_search += 1
                alpha_cur = line_search_obj.next_step(y_prev, )
                step_cur *= alpha_cur / numpy.sqrt(alpha2)

