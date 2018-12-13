import numpy as np

#####################################################################
# loss function
def Lossl2(x, y):
#    xx, yy = x.reshape(-1), y.reshape(-1)
#    f = np.sum((xx - yy)**2)/x.shape[0]
#    g = 2 * (xx - yy) / x.shape[0]

    f = np.sum((x - y)**2)/x.shape[0]
    g = 2 * (x - y) / x.shape[0]
    return f, g


def Cross_entropy_with_sigmoid(x, y):
    # x is a real number, while y is 0 or 1    
    px = 1.0 /(np.exp(-x) + 1)
    py = y.reshape((-1,1))
    
    f = np.sum(- py * np.log(px) - (1 - py) * np.log (1- px))/x.shape[0]
    g = - py * (1 - px) + (1 - py) * px
    return f, g


