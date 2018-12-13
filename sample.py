def LinearSamples(n, A, b, noise = 0):
    d1, d2 = A.shape
    x = np.random.rand(n, d1)
    y = x.dot(A) + b + noise * np.random.randn(n, d2)
    return x, y




def getSamplesX1(n, noise = 3):
    t = 20*np.random.rand(n)
    x = np.empty((n,2))
    x[:,0] = t * t * np.cos(3*t) + noise * np.random.randn(n)
    x[:,1] = t * t * np.sin(3*t) + noise * np.random.randn(n)
    return x

def getSamplesX2(n, noise = 3):
    t = 600*np.random.rand(n)-300
    x = np.empty((n,2))
    x[:,0] = t + noise * np.random.randn(n)
    x[:,1] =  t * np.cos (t/30) + noise * np.random.randn(n)
    return x


def circleSamples(n, center, r, noise= 0):
    t = 2 * np.pi*np.random.rand(n)
    x = np.empty((n,2))
    x[:,0] = center[0] + np.cos(t) * (1 + noise * np.random.randn(n))
    x[:,1] = center[1] + np.sin(t) * (1 + noise * np.random.randn(n))
    return x


def getSamplesX(n,p=0.5, noise = 3):
    n1 = int(p*n)
    n2 = n - n1
    x1 = np.hstack((getSamplesX1(n1, noise),np.zeros((n1,1))))
    x2 = np.hstack((getSamplesX2(n2, noise),np.ones((n2,1))))
    x = np.vstack((x1,x2))
    np.random.shuffle(x)
    return x[:,:2], x[:,2]



def binaryClassification(n, c1, c2, p, noise):
    n1 = int(p*n)
    n2 = n - n1
    x1 = np.hstack((c1(n1, noise),np.zeros((n1,1))))
    x2 = np.hstack((c2(n2, noise),np.ones((n2,1))))
    x = np.vstack((x1,x2))
    np.random.shuffle(x)
    return x[:,:2], x[:,2]


