import numpy as np
import matplotlib.pyplot as plt

def getSample(n, x0 = None, ar = np.array([]), ma = np.array([]), c = 0, sig= 1):
    p, q = len(ar), len(ma)
    if x0 is None: x0 = np.zeros(p)
    e = sig * np.random.randn(n+q)
    x = np.empty(n + p)
    print(p)
    x[:p] = x0[-p:]
    for i in range(n):
        x[p+i] = c + e[i+q] + x[i:i+p].dot(ar) + e[i:i+q].dot(ma)
    return x[p:]
        

#x = getSample(500, x0= np.array([0,0,0]), ar = np.array([-0.1,-0.5,1.3,]), ma = np.array([0.5, -0.2]),sig=0.3)
x = getSample(500, ar = np.array([0.99]),sig=0.02, c= -0.05)


import network as nw
from loss import Lossl2
from optimizer import Gradientdescent
N_train, N_test = 400, 50

x_train, x_test = x[:N_train], x[N_train:N_train+N_test]
y_train, y_test = x[1:N_train+1], x[1+N_train: 1+N_train+N_test]

#layers = [ nw.RnnLayer(1,5, activation="tanh"),
#         nw.RnnLayer(5,2, activation="tanh"),
#        nw.RnnLayer(2,1, activation=None)]

#layers = [ nw.RnnLayer(1,2, activation=None), nw.TempAffineLayer(2,1) ]
layers = [ nw.LSTMLayer(1,1), nw.TempAffineLayer(1,1) ]


net = nw.NeuralNetwork(layers, Lossl2, optimizer = Gradientdescent(alpha = 0.03, decay_rate = 0.99, decay_step = 200))
net.train(x_train, y_train, max_iter = 1000, print_every = 100, batch_size = None)

layers[0].print()
print("Wx+Wh=", layers[0].Wx + layers[0].Wh)

x_pre = np.zeros(N_train+N_test)

x_pre[:N_train] = net.predict((x[:N_train]).reshape(-1)).reshape(-1)
for i in range(N_test):
    x_pre[i+N_train] = net.predict(x[i+N_train - 1]).reshape(-1)

plt.plot(x_pre, c='red')
plt.plot(x[1:1+N_train+N_test], c='black')
plt.show()

