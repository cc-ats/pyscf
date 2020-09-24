import numpy

class OptimizeAlgorithm(object):
    pass

class LineSearch(OptimizeAlgorithm):
    def __init__(self, scale = 0.1, max_line_search=15):
        self.scale = scale

        self._alpha_left  = None
        self._g_left      = None
        self._y_left      = None
        self._alpha_right = None
        self._g_right     = None
        self._y_right     = None

        self.max_line_search = max_line_search

    def reset(self):
        self._alpha_left  = None
        self._g_left      = None
        self._y_left      = None
        self._alpha_right = None
        self._g_right     = None
        self._y_right     = None


    def next_step(self, iter_line_search, y_prev, alpha_prev, g_prev, y_next, alpha_next, g_next, is_wolfe1, is_wolfe2):
        if iter_line_search == 1:
            self._alpha_left  = alpha_prev
            self._g_left      = g_prev
            self._y_left      = y_prev
            self._alpha_right = None
            self._g_right     = None
            self._y_right     = None
        elif iter_line_search < 0:
            RuntimeError("Bad input for LineSearchOptimizer")

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

            if alpha_new > self._alpha_right:
                is_line_search_step_fail = True
            elif alpha_new < self._alpha_left + scale * alpha_len:
                alpha_new = self._alpha_left + scale * alpha_len
            elif alpha_new > self._alpha_right - scale * alpha_len:
                alpha_new = self._alpha_right - scale * alpha_len

        elif not is_wolfe2:
            alpha_new = (alpha_next*self._g_left + self._alpha_left*g_next)/(self._g_left-g_next)
            alpha_len = alpha_next - self._alpha_left

            if alpha_new < self._alpha_left or alpha_new > self._alpha_left + 10.0 * alpha_len:
                alpha_new = self._alpha_left + 10.0 * alpha_len
            elif alpha_new < self._alpha_left + 3.0 * alpha_len:
                alpha_new = self._alpha_left + 3.0 * alpha_len
            
            self._alpha_left = alpha_next
            self._g_left     = g_next
            self._y_left     = y_next

            if alpha_new > (1 - scale) * self._alpha_right + scale * alpha_next:
                alpha_new = (1 - scale) * self._alpha_right + scale * alpha_next

        assert not is_line_search_step_fail
        alpha_next = alpha_new

        

class DogLegSearch(OptimizeAlgorithm):
    def __init__(self):
    def next_step(y_prev, )

def next_step(f, gradf, x, alpha=5.0, beta=1e-4, sigma=0.1, max_line_search=15):
    step     = None
    
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
                next_line_search_step(iter_line_search, )
                step_cur *= alpha_cur / numpy.sqrt(alpha2)

