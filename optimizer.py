import numpy as np

class Optimizer(object):

    def __init__(self, f, g, x0, step_size = 0.1, noise_level = 0, max_iter =200, decay = 0):
        self.f = f
        self.g = g
        self.x = x0

    #######################################################################################
    # stop type of the algorithm
    # "type":"max_iter", "max_iter" : int 
    # "type":"oscillate", "max_iter" : int, "eps": float
    # "type":"FOSP", "max_iter" : int, "eps": float
    #######################################################################################
        
        #self.max_iter = stop["max_iter"]
        self.max_iter = max_iter

        #if stop["type"] != "max_iter"
        #    self.eps = stop["eps"]
        
        self.n_iter = 0
        self.step_size = step_size
        self.noise_level = noise_level
        self.his = [x0]
        self.decay = decay

    def minimize(self, history = False, print_every = None):

        x = self.x
        while self.stop() != True:
            x = self.iterate(x)
            self.n_iter += 1
            self.step_size *=  1 - self.decay
            if history:
                (self.his).append(x)
            if print_every != None and self.n_iter % print_every == 0:
                print(x)

        if history:
            return x, self.his

        return x

        
    def stop(self):
        if self.n_iter > self.max_iter:
            return True
        


    def iterate(self, x):
        pass


    def plotIter(self):
        pass


class GradientDescent(Optimizer):

    def iterate(self, x):
        s = self.step_size + self.noise_level * np.random.randn()
        return x - s * (self.g(x))
        

class GradientSignDescent(Optimizer):

    def iterate(self, x):
        s = self.step_size + self.noise_level * np.random.randn()
        return x - s * np.sign(self.g(x))
        
   
