import numpy as np
import numpy.random as npr
import numpy.linalg as npla
import scipy as sp
import scipy.stats as sps
import matplotlib.pyplot as plt

##################
#DEFINE FUNCTIONS#
##################
#Define a 2d normal mixture distributions with k components each having means M (np array of size by 2 by k), covariances C (np array of size k by 2 by  2)
def norm_mixture_post(x, M, C):
    k = M.shape[1]
    density = 0
    for i in range(0,k):
        density += sps.multivariate_normal.pdf(x,M[:,i],C[i,:,:])

    return(density)

def rad_calc(x,samps,N):
    if N > np.shape(self.S)[0]:
        print('Not possible!')
        return()

    radii = np.sort(npla.norm(self.S-x,ord=2,axis=1))
    R = radii[N]
    return(R)

def interpolate_density(x, s, r_def, r):  
    B = s[0]
    fB = s[1]

    dim = B.shape()[0]
    N = B.shape()[1]
    
    W = np.sqrt( [ min(1, (1 - ( (npla.norm(b-x) - r_def ) / (r_def) )**3) **3 ) for b in B.T ] )    
    W = np.diag(W)

    phi = np.zeros([N,2*dim+1])
    phi[:,0] = np.ones(N)
    phi[:,1:(dim+1)] = B
    phi[:,(dim+1):(2*dim+1)] = B**2
    q, r = npla.qr(np.dot(W,phi), mode='complete')
    q = q[:,0:r.shape[1]]
    r = r[0:r.shape[1],:]
    
    Z = np.dot(npla.inv(r), q.T)
    Z = np.dot(Z, np.dot(W,fB))

    return(z)
    
############
#CODE BEGIN#
############
k = 5
M = np.array([[1,-1], [1,1], [-1,1], [-1,-1]])
C = np.array([np.eye(2) for unused_index in range(0,k)])

n_S = 1e3
n_S = int(n_S)
S = [np.array([npr.uniform(-10,10,2) for unloved_index in range(0,n_S)]),[]]
for i in range(0,n_S):
    S[1].append( norm_mixture_post(S[0][i], M[i], C[i,:,:]) )
S[1] = np.array(S[1])




