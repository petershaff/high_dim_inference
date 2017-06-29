import numpy as np
import numpy.linalg as npla
import numpy.random as npr

import scipy as sp
import scipy.linalg as spla
import scipy.stats as sps
import scipy.interpolate as spi

import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sklearn.neighbors as skn

import math
import itertools as it
import time

#Evaluate numerical gradient of arbitrary scalar function at x
def grad(x, func, grad_grid_width = 1e-2):
    d = len(x)

    V = lambda x: func(x)
    
    grid_pts = np.array([np.linspace( xi - grad_grid_width/2., xi + grad_grid_width/2., 5) for xi in x]  )

    grid_vals = np.zeros([d,5])
    for i in range(0,d):
        for j in range(0,5):
            hold = grid_pts[:,2].copy()
            hold[i] = grid_pts[i,j]
            grid_vals[i,j] = V( hold )

    grads = np.array(np.gradient(grid_vals, grad_grid_width, axis = 1))
    
    return(grads[:,2])
    
#1 step of the leapfrog integration with potential energy function V under quadratic kinetic energy
def lpfrg_pshfwd(x0, h, V, M):
    d = int(len(x0)/2.)
    [q0, p0] = [ x0[0:d], x0[ d:(2*d + 1) ] ]

    pt = p0 - (h/2.)*grad(q0,V)
    qt = q0 + np.dot(h*npla.inv(M), pt)
    pt = pt - (h/2.)*grad(qt,V)
    
    return( np.array( list(qt) + list(pt) ) )


#TEST DENSITY
class test_post:
    def __init__(self, A, B, C, P):
        self.a = A
        self.b = B
        self.c = C
        self.p = P

    def evaluate(self, z):
        [x,y] = z
        evaluate = np.exp( -1 * self.p * ( self.a * x**2 * y**2  +  x**2  +  y**2  -  self.b * x * y  -  self.c * x  - self.c * y ) )

        return(evaluate)
        
    def grad(self, z):
        [x,y] = z
        grad = np.zeros(2)
        grad[0] = -1*np.exp( -1 * self.p * ( 2 * self.a * x * y**2  +  2 * x  -  self.b * y  -  self.c ) )
        grad[1] = -1*np.exp( -1 * self.p * ( 2 * self.a * x**2 * y  +  2 * y  -  self.b * x  -  self.c ) )

        return(grad)

#def test_dense(x,y):
#    M = np.array([ [3.,3.], [0.,0.], [-3., 3.] ])
#    vals = np.array([sps.multivariate_normal.pdf(np.array([x,y]).T, m, np.eye(2)) for m in M]).T
#
#    if len(vals.shape) == 1:
#        return( sum(vals) )
#
#    else:
#        return( np.sum(vals, 1) )
#


#METHOD 1: LOCAL QUAD REGRESSION
class local_quad_reg:
    def __init__(self, S, fS):
        #Save the provided data internally, take some useful measurements
        self.S = S
        self.fS = fS
        self.d = self.S.shape[1]
        self.Ndef = int((2*self.d+1)*(self.d+2)/2)
        self.N = int(np.ceil(np.sqrt(self.d)*self.Ndef))
        self.n = self.S.shape[0]
      
    def evaluate(self, x):
        self.interpolate_density(x)
        val = np.dot( self.Z, np.hstack([ 1, x, x**2 ]) )

        return( val )

    def grad(self, x):
        self.interpolate_density(x)
        grad = np.zeros(self.d)
        for i in range(1, self.d+1):
            mask = np.zeros(1 + 2*self.d)
            mask[i] = 1
            mask[i+self.d] = 1
            grad[i-1] = np.dot(self.Z, np.hstack([0, np.ones(self.d), 2*x]) * mask)

        return(grad)
    
    def neighbors(self, x):
        knn = skn.NearestNeighbors(n_neighbors = self.N)
        knn.fit(self.S)
        [self.dists, self.inds] = knn.kneighbors( np.array([x]) )

        knn = skn.NearestNeighbors(n_neighbors = self.Ndef)
        knn.fit(self.S)
        [self.dists_def, self.inds_def] = knn.kneighbors( np.array([x]) )
       
        self.B = self.S[self.inds][0]
        self.fB = self.fS[self.inds][0]

        self.r_def = np.max( self.dists_def )

    def interpolate_density(self, x):
        self.neighbors(x)
        
        self.W = np.sqrt( np.array([ min(1,x) for x in 1 - ( (self.dists[0] - self.r_def ) / self.r_def**3 )**3 ]) )
        self.W = np.diag(self.W)

        self.phi = np.zeros([ self.N, 2*self.d + 1 ])
        self.phi[:,0] = np.ones(self.N)
        self.phi[:, 1:( self.d+1 )] = self.B
        self.phi[:, ( self.d+1 ):( 2*self.d+1 )] = self.B**2
        q, r = npla.qr(np.dot( self.W, self.phi), mode='complete')
        q = q[:,0:r.shape[1]]
        r = r[0:r.shape[1],:]

        self.Z = np.dot(npla.inv(r), q.T)
        self.Z = np.dot(self.Z, np.dot( self.W, self.fB))
        
        
#METHOD 2: RADIAL BASIS FUNCTIONS - THIN PLATE SPLINES
#Define a class that fits a low-rank thin plate spline to scattered data sites S and function values fS with wiggliness penalized by lambda to derivative order m.  Follows Wood, 2002.
class thin_plate:
    def __init__(self, S, fS, m = 2, lam = 1e-13, lr_tol = .999):
        self.timer = {}
        
        #Save the provided data internally, take some useful measurements
        self.S = S
        self.fS = fS
        self.m = m
        self.lam = lam
        self.lr_tol = lr_tol
        
        self.n = self.S.shape[0]
        self.d = self.S.shape[1]

        #Compute the matrix T of multivariate monomials of degree <= m, in d variables (note: wood gives wrong formula for self.M, see: stars and bars problem)
        self.curr_time = time.time()
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

        self.timer['Build T'] = time.time() - self.curr_time
        self.curr_time = time.time()

        #Compute the matrix E of RBF values between points
        if self.d%2 == 0:
            self.eta = lambda x: np.log(x)*pow(x,2*self.m-self.d)*pow(-1,self.m+1+0.5*self.d)/(pow(2,2*self.m - 1)*pow(np.pi,0.5*self.d)*math.factorial(self.m-1)*math.factorial(self.m-0.5*self.d))
            self.eta_prime = lambda x: pow(x,2*self.m-self.d - 1)*( 1 + np.log(x)*(2*self.m-self.d) )*pow(-1, self.m + 0.5*self.d +1) / (pow(2, 2*self.m-1)*pow(np.pi, 0.5*self.d)*math.factorial( self.m - 1 )*math.factorial( self.m - 0.5*self.d) )
            
        elif self.d%2 == 1:
            self.eta = lambda x: pow(x,2*self.m-d)*math.gamma(0.5*self.d - self.m)/(pow(2,2*self.m)*pow(np.pi,0.5*self.d)*math.factorial(self.m-1))
            self.eta_prime = lambda x: (2*self.m-d)*pow(x,2*self.m-d-1)*math.gamma(0.5*self.d - self.m)/(pow(2,2*self.m)*pow(np.pi,0.5*self.d)*math.factorial(self.m-1))

        self.E = np.zeros( [self.n, self.n] )
        
        for i in range(0,self.n):
            for j in range(0,self.n):
                if i == j:
                    self.E[i,j] = 0

                else:
                    self.E[i,j] = self.eta( npla.norm(self.S[i,:]-self.S[j,:]) )

        self.timer['Build E'] = time.time() - self.curr_time
        self.curr_time = time.time()
        #Calculate low-rank approx of E
        self.approx_E()
        self.timer['Low-rank approx E'] = time.time() - self.curr_time
        self.curr_time = time.time()

        #Calculate null space of T' U_k
        self.null_space_T()
        self.timer['Null space of T'] = time.time() - self.curr_time
        self.curr_time = time.time()

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
        
        self.timer['Solving LS problem'] = time.time() - self.curr_time

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

        return( np.sum( np.dot(E, self.delta) + np.dot(T, self.a) ) )

    def grad(self, x):
        grad = np.zeros(self.d)

        for i in range(0, self.d):
            grad_E = np.array([ self.eta_prime(npla.norm(x - S[i])) * 2*(x[i]-s[i])/npla.norm(x - s) for s in self.S])

            grad_T = np.zeros( int(self.M) )
            j = 0
            for mu in range(0, self.m+1):
                hold = self.partitions(mu, self.d)
                for k in range(0, int( sp.special.binom(mu + self.d - 1, mu) ) ):
                    exp = next(hold)

                    if exp[i] != 0:
                        exp[i] = exp[i] - 1
                        t = map( lambda (x,y): pow(x,y), zip(x,exp) )
                        t[i] = (exp[i]+1)*t[i]
                        t = sum(t)

                    else:
                        t = 0
                
                    grad_T[j] = t
                    j += 1
              
            grad[i] = np.dot(grad_E, self.delta) + np.dot(grad_T, self.a)

        return(grad)

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
        low_rank_ind = part_sums < self.lr_tol

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

#Implements a Thin Plate Spline as outlined by Ji and Kim (2013)
class thin_plateV2:
    def __init__(self, S, fS, m=2, lam = 1e-13, lr_tol = .999):
        self.timer = {}
        
        #Save the provided data internally, take some useful measurements
        self.S = S
        self.fS = fS
        self.lam = lam
        self.lr_tol = lr_tol
        
        self.n = self.S.shape[0]
        self.d = self.S.shape[1]

        #Compute the matrix T of multivariate monomials of degree <= m, in d variables (note: wood gives wrong formula for self.M, see: stars and bars problem)
        self.curr_time = time.time()
        self.m = self.d + 1
        self.M = sp.special.binom( self.m + self.d, self.d)

        if self.M >= self.n:
            raise ValueError('Not enough points in data set for this degree of smoothness, add more points or choose lower m')

        self.T = np.zeros( [ int(self.n) , int(self.m) ] )

        for i in range(0, self.n):
            self.T[i,:] = np.concatenate([ np.array([1]), self.S[i] ])
            
        self.timer['Build T'] = time.time() - self.curr_time
        self.curr_time = time.time()

        #Compute the matrix E of RBF values between points
        if self.d%2 == 0:
            self.eta = lambda x: np.log(x)*pow(x,2*self.m-self.d)*pow(-1,self.m+1+0.5*self.d)/(pow(2,2*self.m - 1)*pow(np.pi,0.5*self.d)*math.factorial(self.m-1)*math.factorial(self.m-0.5*self.d))
            self.eta_prime = lambda x: pow(x,2*self.m-self.d - 1)*( 1 + np.log(x)*(2*self.m-self.d) )*pow(-1, self.m + 0.5*self.d +1) / (pow(2, 2*self.m-1)*pow(np.pi, 0.5*self.d)*math.factorial( self.m - 1 )*math.factorial( self.m - 0.5*self.d) )
            
        elif self.d%2 == 1:
            self.eta = lambda x: pow(x,2*self.m-d)*math.gamma(0.5*self.d - self.m)/(pow(2,2*self.m)*pow(np.pi,0.5*self.d)*math.factorial(self.m-1))
            self.eta_prime = lambda x: (2*self.m-d)*pow(x,2*self.m-d-1)*math.gamma(0.5*self.d - self.m)/(pow(2,2*self.m)*pow(np.pi,0.5*self.d)*math.factorial(self.m-1))

        self.E = np.zeros( [self.n, self.n] )
        
        for i in range(0,self.n):
            for j in range(0,self.n):
                if i == j:
                    self.E[i,j] = 0

                else:
                    self.E[i,j] = self.eta( npla.norm(self.S[i,:]-self.S[j,:]) )

        self.timer['Build E'] = time.time() - self.curr_time
        self.curr_time = time.time()

        #Now reformulate and solve the approximate problem
        self.C = np.bmat([ [self.E, self.T], [np.transpose(self.T), np.zeros([self.m, self.m ]) ] ])
        self.y = np.concatenate([ self.fS, np.zeros([ self.m ]) ])
        
        self.C_inv = npla.inv(self.C)
        self.gamma = np.dot( self.C_inv, self.y )
        
        self.delta = self.gamma[:, :self.n]
        self.a = self.gamma[:, self.n:]

    def evaluate(self, x):
        E = np.array([ self.eta(npla.norm(x - s)) for s in self.S])
        T = np.concatenate([ np.array([1]), x ])
        
        return( np.sum( np.dot(E, np.transpose(self.delta) ) + np.dot( T, np.transpose(self.a) ) ) )

    def grad(self, x):
        grad = np.zeros(self.d)

        for i in range(0, self.d):
            grad_E = np.array([ self.eta_prime(npla.norm(x - S[i])) * 2*(x[i]-s[i])/npla.norm(x - s) for s in self.S])

            grad_T = np.zeros( int(self.M) )
            j = 0
            for mu in range(0, self.m+1):
                hold = self.partitions(mu, self.d)
                for k in range(0, int( sp.special.binom(mu + self.d - 1, mu) ) ):
                    exp = next(hold)

                    if exp[i] != 0:
                        exp[i] = exp[i] - 1
                        t = map( lambda (x,y): pow(x,y), zip(x,exp) )
                        t[i] = (exp[i]+1)*t[i]
                        t = sum(t)

                    else:
                        t = 0
                
                    grad_T[j] = t
                    j += 1
              
            grad[i] = np.dot(grad_E, self.delta) + np.dot(grad_T, self.a)

        return(grad)

################################
#TEST AND PLOT THE APPROXIMANTS#
################################

#Define test density
A = 1
B = 10
C = 1
P = 0.05
test_dense = test_post(A,B,C,P)


#Get interpol/approximating sets
x = npr.uniform(-15,15,300)
y = npr.uniform(-15,15,300)

S = np.array([x,y]).T
fS = test_dense.evaluate( [x,y] )

#Time the initialization of both methods
print('Init LQR')
start_time = time.time()
lqr_test = local_quad_reg(S, fS)
lqr_init_time = time.time() - start_time

print('Init TPS')
start_time = time.time()
tps_test = thin_plateV2(S, fS, lam = 1e-7, lr_tol = 1.2)
tps_init_time = time.time() - start_time

def lqr_dense(x,y):
    return( lqr_test.evaluate( np.array([x,y]) ) )

def tps_dense(x,y):
    return( tps_test.evaluate( np.array([x,y]) ) )

#Testing grid
xi = np.linspace(-6,6,100)
yi = np.linspace(-6,6,100)

#Time the calculation of interp/approx vals and grad on testing grid
#print('Calculating LQR interp density')
#start_time = time.time()
#lqr_d = np.reshape([lqr_dense(x,y) for x in xi for y in yi], [len(xi), len(yi)])
#lqr_gd = np.reshape([ npla.norm( grad([x,y], lqr_test.evaluate) - grad([x,y], test_dense.evaluate) ) for x in xi for y in yi ], [len(xi), len(yi)])
#lqr_time = time.time() - start_time

print('Calculating TPS interp density')
start_time = time.time()
tps_d = np.reshape([tps_dense(x,y) for x in xi for y in yi], [len(xi), len(yi)])
tps_gd = np.reshape([ npla.norm( grad([x,y], tps_test.evaluate) - grad([x,y], test_dense.evaluate) ) for x in xi for y in yi ], [len(xi), len(yi)])
tps_time = time.time() - start_time

#For error calcs
print('Calculating true density')
start_time = time.time()
d = np.reshape([test_dense.evaluate([x,y]) for x in xi for y in yi], [len(xi), len(yi)] )
gd = np.reshape([ npla.norm( grad([x,y], test_dense.evaluate) ) for x in xi for y in yi ], [len(xi), len(yi)] )
true_time = time.time() - start_time

#Check max val and grad error
lqr_err = np.max(np.abs( (lqr_d - d))) / np.max( np.abs(d) )
tps_err = np.max(np.abs( (tps_d - d))) / np.max( np.abs(d) )

lqr_grad_err = np.max(np.abs( lqr_gd - gd )) / np.max( gd )
tps_grad_err = np.max(np.abs( tps_gd - gd )) / np.max( gd )

#Evaluate how lambda effects err
#print('Testing lambda vs. err')
#lams = [1e-10, 1e-7,1e-4, 1e-1, 1, 10, 100]
#lam_err = []
#lam_grad_err =[]
#for l in lams:
#    print(l)
#    tps_lam_test = thin_plate(S, fS, lam = l, lr_tol = 1.2)

#    def tps_lam_dense(x,y):
#        return( tps_test.evaluate( np.array([x,y]) ) )

#    tps_d_lam = np.reshape([tps_lam_dense(x,y) for x in xi for y in yi], [len(xi), len(yi)])
#    tps_gd_lam = np.reshape([ npla.norm( grad([x,y], tps_test.evaluate) - grad([x,y], test_dense.evaluate) ) for x in xi for y in yi ], [len(xi), len(yi)])

#    lam_err.append( np.max(np.abs( (tps_d_lam - d))) / np.max( np.abs(d) ) )
#    lam_grad_err.append( np.max(np.abs( tps_gd_lam - gd )) / np.max( gd ) )


#Test pushforward error
print('Testing pushforward err')
test_n = 100
lqr_ham_err = []
tps_ham_err = []

def lqr_pot(z):
    [x,y] = z
    return( -1*np.log( np.abs(lqr_dense(x,y)) ) )

def tps_pot(z):
    [x,y] = z
    return( -1*np.log( np.abs(tps_dense(x,y)) ) )

def test_pot(z):
    return( -1*np.log( np.abs(test_dense.evaluate(z)) ) )

def path_dist(x):
    dist = 0
    
    for i in range(1, x.shape[0]):
        dist += npla.norm( x[i] - x[i-1] )

    return(dist)
        

h = 1e-2
for unused_index in range(0,test_n):
    #steps = np.random.choice(range(1,3))
    steps = np.random.choice( range(10,100) )

    lqr_x0 = np.array([ np.random.uniform(-6,6) for unused2 in range(0,4) ])
    tps_x0 = lqr_x0.copy()
    test_x0 = tps_x0.copy()

    lqr_path = [lqr_x0]
    tps_path = [tps_x0]
    test_path = [test_x0]

    
    for i in range(0,steps):
        print([unused_index,i, steps])
        lqr_x0 = lpfrg_pshfwd(lqr_x0, h, lqr_pot, np.eye(2))
        tps_x0 = lpfrg_pshfwd(lqr_x0, h, tps_pot, np.eye(2))
        test_x0 = lpfrg_pshfwd(lqr_x0, h, test_pot, np.eye(2))

        lqr_path.append(lqr_x0)
        tps_path.append(tps_x0)
        test_path.append(test_x0)

    lqr_ham_err.append( sum( npla.norm( np.array(lqr_path) - np.array(test_path) , axis = 1 ) ) / path_dist( np.array(test_path) ) )
    tps_ham_err.append( sum( npla.norm( np.array(tps_path) - np.array(test_path) , axis = 1 ) ) / path_dist( np.array(test_path) ) )

print( np.mean(lqr_ham_err) )
print( np.mean(tps_ham_err) )


################
#PLOTTING STUFF#
################
#[xi, yi] = np.meshgrid(xi, yi)
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(xi,yi, lqr_d)
#plt.show()

#plt.contour(xi,yi, np.abs(d-di)/np.abs(np.max(d)))
#plt.show()



#######
#CRUFT#
#######
#plt.plot(xi,d)
#plt.plot(xi,di)
#plt.show()

#ax.scatter(S[:,0], S[:,1], fS)
#xi = yi = np.linspace(-10,10,100)
#di = np.reshape([rbi(x,y) for x in xi for y in yi], [len(xi),len(yi)])
#x_grid = np.reshape([x for x in xi for y in yi], [len(xi),len(yi)])
#y_grid = np.reshape([y for x in xi for y in yi], [len(xi),len(yi)])

#ax.scatter(xi,yi,di)
#plt.show()
#def frob_err(k):
#    d_lr = np.diag( test.D[:k] )
#    u_lr = test.U[:,:k]
#    e_lr = np.dot( u_lr, np.dot(d_lr,u_lr.T) )
#    return( npla.norm(test.E - e_lr, 'fro')/npla.norm(test.E, 'fro') )
#

