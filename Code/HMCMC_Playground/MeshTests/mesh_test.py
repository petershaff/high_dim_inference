import numpy as np
import numpy.linalg as npla
import numpy.random as npr

import scipy as sp
import scipy.linalg as spla
import scipy.stats as sps
import scipy.interpolate as spi

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sklearn.neighbors as skn

import math
import itertools as it

def grad(x, V = lambda x: x, grad_grid_width = 1e-2):
    d = len(x)
    
    grid_pts = np.array([np.linspace( xi - grad_grid_width/2., xi + grad_grid_width/2., 5) for xi in x]  )
    grid_vals = V(grid_pts)

    grads = np.array(np.gradient(grid_vals, grad_grid_width, edge_ord = 2, axis = 1))
    
    return(grads[:,2])    
    

def lpfrg_pshfwd(x0, h, V, M):
    d = int(len(xt)/2.)
    [q0, p0] = [ xt[0:d], x0[ (2*d + 1):d ] ]

    pt = p0 - (h/2.)*grad(q0,V)
    qt = q0 + h*npla.inv(M)*pt
    pt = pt - (h/2.)*grad(qt,V)

    return( np.array( list(qt) + list(pt) ) )

#TEST DENSITY
def test_dense(x,y):
    M = np.array([ [5.,5.], [0.,0.], [-5., 5.] ])
    vals = np.array([sps.multivariate_normal.pdf(np.array([x,y]).T, m, np.eye(2)) for m in M]).T

    if len(vals.shape) == 1:
        return( sum(vals) )

    else:
        return( np.sum(vals, 1) )

#METHOD 2: RADIAL BASIS FUNCTIONS - THIN PLATE SPLINES
#Define a class that fits a sparse thin plate spline to scattered data sites S and function values fS with wiggliness penalized to derivative order m.  Follows Wood, 2002. 
class thin_plate:
    def __init__(self, S, fS, m = 2, lam = 1e-13):
        #Save the provided data internally, take some useful measurements
        self.S = S
        self.fS = fS
        self.m = m
        self.lam = lam

        self.n = self.S.shape[0]
        self.d = self.S.shape[1]

        #Compute the matrix T of multivariate monomials of degree <= m, in d variables (note: wood gives wrong formula for self.M, see: stars and bars problem)
        self.M = sp.special.binom( self.m + self.d, self.d)

        if self.M >= self.n:
            raise ValueError('Not enough points in data set for this degree of smoothness, add more points or choose lower m')

        self.T = np.zeros( [ int(self.n) , int(self.M) ] )

        for i in range(0, self.n):
            j = 0
            
            for mu in range(0, self.m+1):
                hold = self.partitions(mu, self.d)

                for k in range(0, int( sp.special.binom(mu + self.d - 1, mu) ) ):
                    exp = next(hold)
                    x = self.S[i]

                    t = sum( map( lambda (x,y): pow(x,y), zip(x,exp) ) )

                    self.T[i,j] = t 
                    j += 1

        #Compute the matrix E of RBF values between points
        if self.d%2 == 0:
            self.eta = lambda x: np.log(x)*pow(x,2*self.m-self.d)*pow(-1,self.m+1+0.5*self.d)/(pow(2,2*self.m - 1)*pow(np.pi,0.5*self.d)*math.factorial(self.m-1)*math.factorial(self.m-0.5*self.d))

        elif self.d%2 == 1:
            self.eta = lambda x: pow(x,2*self.m-d)*math.gamma(0.5*self.d - self.m)/(pow(2,2*self.m)*pow(np.pi,0.5*self.d)*math.factorial(self.m-1))

        self.E = np.zeros( [self.n, self.n] )
        
        for i in range(0,self.n):
            for j in range(0,self.n):
                if i == j:
                    self.E[i,j] = 0

                else:
                    self.E[i,j] = self.eta( npla.norm(self.S[i,:]-self.S[j,:]) )

        #Calculate low-rank approx of E
        self.approx_E()

        #Calculate null space of T' U_k
        self.null_space_T()

        #Now reformulate and solve the approximate problem
        self.X = np.dot( self.U_lr, np.dot(self.D_lr, self.Z) )
        self.X = np.bmat([self.X, self.T])

        self.W = np.dot( self.Z.T, np.dot( self.D_lr, self.Z))
        l = int(self.k - self.M)
        M = int(self.M)
        self.W = np.bmat([[self.W,np.zeros([l,M])],[np.zeros([M,l]),np.zeros([M,M])]])

        self.sol = sp.optimize.minimize( lambda z: npla.norm(self.fS - np.dot(self.X,z)) + self.lam*npla.norm(np.dot(z, np.dot(self.W,z).T)), np.random.rand(self.k))
        self.b = self.sol.x
        
        self.delta = np.dot(self.U_lr, np.dot(self.Z, self.b[:l]))
        self.a = self.b[l:]

    def evaluate(self, x):
        E = np.array([ self.eta(npla.norm(x - s)) for s in self.S])
        T = np.zeros( int(self.M) ) 

        j = 0
        for mu in range(0, self.m+1):
            hold = self.partitions(mu, self.d)
            for k in range(0, int( sp.special.binom(mu + self.d - 1, mu) ) ):
                exp = next(hold)
                t = sum( map( lambda (x,y): pow(x,y), zip(x,exp) ) )

                T[j] = t
                j += 1

        return( np.sum( np.dot(E,self.delta) + np.dot(T,self.a) ) )
                         
    def partitions(self, n, k):
        for c in it.combinations(range(n+k-1), k-1):
            yield [b-a-1 for a, b in zip((-1,)+c, c+(n+k-1,))]

    def approx_E(self):
        [self.D, self.U] = npla.eig(self.E)

        sort_inds = np.argsort( np.abs(self.D) )[::-1]

        self.U = self.U[:, sort_inds]
        self.D = self.D[sort_inds]

        part_sums = np.array([ sum(np.abs(self.D)[:i]) for i in range(0,len(self.D))])
        part_sums = part_sums/sum(np.abs(self.D))
        low_rank_ind = part_sums < .995

        self.k = np.sum(low_rank_ind)
        while self.k <= self.M:
            low_rank_ind[k] = True
            self.k = np.sum(low_rank_ind)
        
        self.D_lr = np.diag(self.D[low_rank_ind])
        self.U_lr = self.U[:,low_rank_ind]

        self.E_lr = np.dot( np.dot(self.U_lr, self.D_lr), self.U_lr.T )
        self.low_rank_err = npla.norm(self.E - self.E_lr, 'fro')/npla.norm(self.E, 'fro')

    def null_space_T(self):
        [self.Q, self.R] = npla.qr( np.dot(self.U_lr.T, self.T), 'complete')
        self.Z = self.Q[:, int(self.M):]

        
x = npr.uniform(-10,10,100)
y = npr.uniform(-10,10,100)

S = np.array([x,y]).T
fS = test_dense(x,y)

test = thin_plate(S,fS)

def frob_err(k):
    d_lr = np.diag( test.D[:k] )
    u_lr = test.U[:,:k]
    e_lr = np.dot( u_lr, np.dot(d_lr,u_lr.T) )
    return( npla.norm(test.E - e_lr, 'fro')/npla.norm(test.E, 'fro') )

#rbi = spi.Rbf(x, y, d, function = 'thin-plate')

def interp_dense(x,y):
    return( test.evaluate( np.array([x,y]) ) )
                                 
xi = np.linspace(-10,10,100)
yi = np.linspace(-10,10,100)
print('Calculating interpolated density')
di = np.reshape([interp_dense(x,y) for x in xi for y in yi], [len(xi), len(yi)])
print('Calculating true density')
d = np.reshape([test_dense(x,y) for x in xi for y in yi], [len(xi), len(yi)])

[xi, yi] = np.meshgrid(xi, yi)
plt.contour(xi,yi, np.abs(d-di)/np.abs(np.max(d)))
plt.show()

#plt.plot(xi,d)
#plt.plot(xi,di)
#plt.show()

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')

#xi = yi = np.linspace(-10,10,100)
#di = np.reshape([rbi(x,y) for x in xi for y in yi], [len(xi),len(yi)])
#x_grid = np.reshape([x for x in xi for y in yi], [len(xi),len(yi)])
#y_grid = np.reshape([y for x in xi for y in yi], [len(xi),len(yi)])

#ax.scatter(xi,yi,di)
#plt.show()
            
#METHOD 1: LOCAL QUAD REGRESSION
#def neighbors(x,S,N):
#    knn = skn.NearestNeighbors(n_neighbors = N)
#    knn.fit(S)
#    [dists, inds] = knn.kneighbors(x)

#    B = S[inds][0]
#    return([inds,dists])

#def interpolate_density(x, S, fS, Ndef, N):  
#    [inds, dists] = neighbors(x,S,N)

#    B = S[inds][0]
#    fB = fS[inds][0]
    
#    dim = B.shape()[0]
#    N = B.shape()[1]
    
#    W = np.sqrt( min(1, (1 - ( (dists - r_def ) / r_def**3 )**3 ) ) )
#    W = np.diag(W)

#    phi = np.zeros([N,2*dim+1])
#    phi[:,0] = np.ones(N)
#    phi[:,1:(dim+1)] = B
#    phi[:,(dim+1):(2*dim+1)] = B**2
#    q, r = npla.qr(np.dot(W,phi), mode='complete')
#    q = q[:,0:r.shape[1]]
#    r = r[0:r.shape[1],:]
    
#    Z = np.dot(npla.inv(r), q.T)
#    Z = np.dot(Z, np.dot(W,fB))

#    return(Z)

#dim = 2
#Ndef = int((2*dim+1)*(dim+2)/2)
#N = int(np.ceil(np.sqrt(dim)*Ndef))

