#####################################################################
# optimization part

class Optimizer():
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.iter_num = 0
    
    def iterate(self):
        pass

class Gradientdescent(Optimizer):
    def iterate(self, x, g):
        self.iter_num += 1

        x -= self.alpha * g
        if self.decay_rate != None and self.iter_num % self.decay_step ==0 :
            self.alpha *= self.decay_rate
        return x
