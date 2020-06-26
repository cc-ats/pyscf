from numpy import empty


class RealTimeStep(object):
    def __init__(self, shape, natoms):
        pass

class RealTimeResult(object):
    pass

    # def load_rt_step_index(self):
    #     return load_rt_step_index(self.chkfile)

    # def load_rt_step(self, step_index):
    #     if hasattr(step_index, '__iter__'):
    #         return [load_rt_step(self.chkfile, istep_index) for istep_index in step_index]
    #     else:
    #         return load_rt_step(self.chkfile, step_index)