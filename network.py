import numpy as np


class NeuralNetwork():
    def __init__(self, layers, Loss, optimizer = None):
        self.layers = layers
        self.Loss = Loss
        self.optimizer = optimizer
        
    def forward(self, X, y = None, predict=False):
        h = X
        for layer in self.layers:
            h = layer.forward(h, predict = predict)          
        if predict: return h 
        f_loss, self.dh = self.Loss(h.reshape(-1), y)
        return f_loss
    
    def backprop(self, update = False):   
        dh = self.dh
        for layer in reversed(self.layers):
            dh = layer.backprop(dh, update= update, optimizer = self.optimizer)
        
    
    def train(self, X, y, max_iter = 2000, print_every = 50, batch_size = None ):
        N = X.shape[0]
        for i in range(max_iter):
            if batch_size != None:
                sample_indices = np.random.randint(N,size = batch_size)
                X_batch, y_batch = X[sample_indices], y[sample_indices]
            else: X_batch, y_batch = X, y
            loss = self.forward(X_batch, y_batch)
            self.backprop(update= True)
            if i % print_every == 0:
                print("loss:",loss)
    
    
    def predict(self, x):
        return self.forward(x, predict =True)
    
    
class DenseLayer():
    def __init__(self, n, m, activation = None):
        self.W = 0.01 * np.random.randn(n,m)
        self.b = np.zeros(m)
        self.activation = activation
        
    def forward(self, x, predict = False):
        self.cacheX = x
        h = x.dot(self.W)+ self.b 
        
        if self.activation == "relu":
            self.cacheH = h.copy()
            h = np.maximum(h,0)

        return h
    
    def backprop(self, dh, update = False, optimizer = None):
        if self.activation == "relu":
            h = self.cacheH
            dh =  dh * (h > 0)
        x = self.cacheX
        dW = x.T.dot(dh)
        db = np.sum(dh, 0)
        
        dx = dh.dot(self.W.T)
        
        if update == True:
            self.W = optimizer.iterate(self.W, dW)
            self.b = optimizer.iterate(self.b, db)   
        return dx
    

class RnnLayer():
    def __init__(self, n, m, N = 1, activation = "tanh", h0 = None):
        self.Wx = 0.01 * np.random.randn(n,m)
        self.Wh = 0.01 * np.random.randn(m,m)
        self.b = np.zeros(m)
        if h0 is None:
            self.h0 = np.zeros((N,m))
        else: self.h0 = h0
        self.activation = activation

    def print(self):
        print("Wx:", self.Wx, "Wh:", self.Wh, "b:", self.b)
        
    def step_forward(self, x, prev_h):
        next_h = np.empty_like(self.h0)
        next_h = x.dot(self.Wx) + prev_h.dot(self.Wh) + self.b

        if self.activation == "tanh":
            next_h = np.tanh(next_h)

        cache = (x, prev_h, next_h)
        return next_h, cache

    def step_backward(self, dnext_h, cache):
        x, prev_h, next_h = cache

        if self.activation == "tanh":
            dtemp = dnext_h * (1 - next_h**2)
        else: dtemp = dnext_h

        dx, dWx, dWh, db = np.empty_like(x), np.empty_like(self.Wx), np.empty_like(self.Wh), np.empty_like(self.b)

     #   dtemp = dtemp.reshape(1,-1)
        dx = dtemp.dot(self.Wx.T)
        dprev_h = dtemp.dot(self.Wh.T)

        dWx = x.T.dot(dtemp)
      #  dWh = prev_h.reshape(-1,1).dot(dtemp)
        dWh = prev_h.T.dot(dtemp)
        db = dtemp.sum(0)
        return dx, dprev_h, dWx, dWh, db

    def forward(self, x, N = 1, predict=False):
        cache={}

        N, T, D = getReshape(x, N)
        x = x.reshape(N,T,D)

        if self.h0.shape[0]< N:
            self.h0  = np.zeros((N,D))
        elif self.h0.shape[0]>N:
            self.h0 = self.h0[-N:,:]

        h = np.zeros((N,T,len(self.b)))

        self.cacheX = x
        for i in range(T):
            prev_h = self.h0 if i == 0 else h[:,i-1,:]
            h[:,i,:], cache[i] = self.step_forward(x[:,i,:], prev_h)

        if predict == True:
            self.h0 = h[:,-1,:]
        self.cache = cache
        return h

    def backprop(self, dh, N = 1, update = False, optimizer = None):
        dx, dh0, dWx, dWh, db, dprev_hi = np.zeros(self.cacheX.shape), np.zeros(self.h0.shape), np.zeros(self.Wx.shape),\
            np.zeros(self.Wh.shape), np.zeros(self.b.shape), np.zeros(self.h0.shape)
     
        N, T, H = getReshape(dh, N)
        dh = dh.reshape(N,T,H)


        for i in range(T-1,-1,-1):
            
            dx[:,i,:], dprev_hi, dWxi, dWhi, dbi = self.step_backward(dh[:,i,:]+dprev_hi, self.cache[i])
            #dx[i], dprev_hi, dWxi, dWhi, dbi = self.step_backward(dh[i]+dprev_hi, self.cache[i])
            dWx += dWxi.reshape(self.Wx.shape)
            dWh += dWhi
            db += dbi
        
        dh0 = dprev_hi


        if update == True:
            self.Wx = optimizer.iterate(self.Wx, dWx)
            self.Wh = optimizer.iterate(self.Wh, dWh)
            self.b = optimizer.iterate(self.b, db)   
        return dx


class LSTMLayer():
    def __init__(self, n, m, N = 1, h0 = None):
        self.Wx = 0.01 * np.random.randn(n,4*m)
        self.Wh = 0.01 * np.random.randn(m,4*m)
        self.b = np.zeros(4*m)
        if h0 is None:
            self.h0 = np.zeros((N, m))
        else: self.h0 = h0
        self.c0 = np.zeros((N, m))

    def print(self):
        print("Wx:", self.Wx, "Wh:", self.Wh, "b:", self.b)
        
    def step_forward(self, x, prev_h, prev_c):
        next_h, next_c = np.empty_like(self.h0), np.empty_like(self.c0)
        N, H = prev_h.shape
        a = (x.dot(self.Wx) + prev_h.dot(self.Wh) + self.b).reshape(N, 4, H)
        i, f, o, g = sigmoid(a[:,0]), sigmoid(a[:,1]), sigmoid(a[:,2]), np.tanh(a[:,3])
        next_c = f * prev_c + i * g
        tanNC = np.tanh(next_c)
        next_h = o * tanNC
        cache = (x, prev_h, prev_c, tanNC, i, f, o, g)
        return next_h, next_c, cache

    def step_backward(self, dnext_h, dnext_c, cache):
        N ,H = dnext_h.shape
        x, prev_h, prev_c, tanNC, i, f, o, g = cache
        dnext_c = dnext_h * o * (1 - tanNC**2) + dnext_c
        dprev_c = dnext_c * f
        di, df, do, dg = dnext_c * g, dnext_c * prev_c, dnext_h * tanNC, dnext_c * i
        da = np.zeros((N,4,H))
        da[:,0], da[:,1], da[:,2], da[:,3] = di * i * (1 - i), df * f  * (1 - f), do * o * (1 - o), dg * (1 - g**2)
        da = da.reshape(N,-1)
        dWx = x.T.dot(da)
        dx = da.dot(self.Wx.T)
        dWh = prev_h.T.dot(da)
        dprev_h = da.dot(self.Wh.T)
        db = da.sum(0)
        return dx, dprev_h, dprev_c, dWx, dWh, db


    def forward(self, x, N =1, predict= False):
        cache={}

        N, T, D = getReshape(x, N)
        x = x.reshape(N, T, D)

        if self.h0.shape[0]< N:
            self.h0  = np.zeros((N,D))
            self.c0  = np.zeros((N,D))
        elif self.h0.shape[0]>N:
            self.h0 = self.h0[-N:,:]
            self.c0 = self.c0[-N:,:]


        H = self.h0.shape[1]
        h = np.zeros((N,T,H))
        prev_c = np.zeros((N, H))

        for i in range(T):
            prev_h = self.h0 if i == 0 else h[:,i-1,:]
            h[:,i,:], prev_c, cache[i] = self.step_forward(x[:,i,:], prev_h, prev_c)
    
        self.cache = cache

        if predict == True:
            self.h0 = h[:,-1,:]
            self.c0 = prev_c

        return h





         


    def backprop(self, dh, N =1, update = False, optimizer = None):
        
        N, T, H = getReshape(dh, N)
        dh = dh.reshape(N, T, H)
        D = self.cache[0][0].shape[1]
        dx, dh0, dWx, dWh, db, dprev_hi, dprev_c = \
                np.zeros((N,T,D)), np.zeros((N,H)), np.zeros((D,4*H)), np.zeros((H,4*H)), np.zeros(4*H), np.zeros((N,H)), np.zeros((N,H))
        
        for i in range(T-1,-1,-1):
            dx[:,i,:], dprev_hi, dprev_c, dWxi, dWhi, dbi = self.step_backward(dh[:,i,:]+dprev_hi, dprev_c, self.cache[i])
            dWx += dWxi
            dWh += dWhi
            db += dbi
            
        dh0 = dprev_hi

        if update == True:
            self.Wx = optimizer.iterate(self.Wx, dWx)
            self.Wh = optimizer.iterate(self.Wh, dWh)
            self.b = optimizer.iterate(self.b, db)   

        return dx


class TempAffineLayer():
    def __init__(self, n, m, N = 1):
        self.W = 0.01 * np.random.randn(n,m)
        self.b = np.zeros(m)

    def print(self):
        print("Wx:", self.Wx, "Wh:", self.Wh, "b:", self.b)
        
    def step_forward(self, x):
        next_h = x.dot(self.W) + self.b
        cache = x
        return next_h, cache

    def step_backward(self, dnext_h, cache):
        x = cache
        dx, dW, db = np.empty_like(x), np.empty_like(self.W), np.empty_like(self.b)
        dx = dnext_h.dot(self.W.T)
        dW = x.T.dot(dnext_h)
        db = dnext_h.sum(0)

        return dx, dW, db

    def forward(self, x, N = 1, predict=False):
        cache={}
        N, T, D = getReshape(x, N)
        x = x.reshape(N,T,D)
        self.cacheX = x
        h = np.zeros((N,T,len(self.b)))

        for i in range(T):
            h[:,i,:], cache[i] = self.step_forward(x[:,i,:])

        self.cache = cache
        return h

    def backprop(self, dh, N = 1, update = False, optimizer = None):
        dx, dW, db = np.zeros(self.cacheX.shape), np.zeros(self.W.shape), np.zeros(self.b.shape)

        N, T, H = getReshape(dh, N)
        dh = dh.reshape(N,T,H)


        for i in range(T-1,-1,-1):
            dx[:,i,:], dWi, dbi = self.step_backward(dh[:,i,:], self.cache[i])
            dW += dWi
            db += dbi
        
        if update == True:
            self.W = optimizer.iterate(self.W, dW)
            self.b = optimizer.iterate(self.b, db)   
        return dx



def sigmoid(x):
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def getReshape(x, N = 1):
    if N == 1:
        lx = len(x.shape)
        if lx == 0:
            T = D = 1
        elif lx == 1:
            T = x.shape[0]
            D = 1
        elif lx == 2:
            T, D = x.shape
        elif lx == 3:
            N, T, D = x.shape
    return N, T, D

