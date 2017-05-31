import numpy as np
import scipy as sp
import numpy.linalg as npla
import scipy.linalg as scla
from scipy import stats as st
from scipy.integrate import ode
import sys
import matplotlib.pyplot as plt

def test_post(b,dat):
    n = len(dat)
    hold = np.zeros(n)
    if b[1] < 0:
        return(0)
    else:
        for i in range(0,n):
            hold[i] = st.multivariate_normal.pdf(dat[i],b[0],b[1]**2)

        return(np.prod(hold))

def SEIR(t,y,parms):
    s = y[0]
    e = y[1]
    i = y[2]
    r = y[3]
    a = parms[0]
    b = parms[1]
    c = parms[2]
    yprime = [-b*s*i, b*s*i - a*e, a*e - c*i,c*i]
    return(yprime)

def fwdMod(parms,t1,dt=.1,y0=[99,0,1,0],t0=0):
    r = ode(SEIR)
    r.set_initial_value(y0,t0).set_f_params(parms)
    output = np.zeros([int(t1/dt),4])
    for t in np.arange(0,int(t1/dt)):
        output[int(t),:] = r.integrate(r.t+dt)
    
    return(output[int(t1/dt)-1,:])

def radCalc(x,samps,N):
    if N > np.shape(samps)[0]:
        print('Not possible!')
        return()

    if np.ndim(samps)==1:
        samps = np.reshape(samps,[np.shape(samps)[0],1])

    radii = np.sort(npla.norm(samps-x,ord=2,axis=1))
    R = radii[N]
    return(R)

def locQuadRegress(p,parmSamps,funcSamps):
    d = max(np.shape([p]))
    Ndef = int((d+1)*(d+2)/2)
    N = int(np.ceil(np.sqrt(d)*(Ndef)))

    #Calculate necessary radii
    print('Getting radii...')
    Rdef = radCalc(p,parmSamps,Ndef)
    R = radCalc(p,parmSamps,N)

    
    if np.ndim(parmSamps)==1:
        parmSamps = np.reshape(parmSamps,[np.shape(parmSamps)[0],1])

    regress_samps = parmSamps[npla.norm(parmSamps-p,ord=2,axis=1) <= R]
    N = len(regress_samps)
    print('Calculating weights...')
    W = np.diag([min(1,(1-((npla.norm(regress_samps[i,:]-p)-Rdef)/(R-Rdef))**3)**3) for i in range(0,np.shape(regress_samps)[0])])
    phi = np.zeros([N,2*d+1])
    phi[:,0] = np.ones(N)
    phi[:,1:(d+1)] = regress_samps
    phi[:,(d+1):(2*d+1)] = regress_samps**2
    
    Y = funcSamps[npla.norm(parmSamps-p,ord=2,axis=1) <= R,:]

    Y = np.dot(W,Y)
    q,r = npla.qr(np.dot(phi.T,np.dot(W,phi)))
    print('Regressing...')
    Z = np.dot(npla.inv(r),np.dot(q.T,np.dot(phi.T,Y)))

    return(np.dot(np.append(1,np.append(p,p**2)),Z))
 
#Test basic function-response regression
#N = 100
#b_samps = np.random.uniform(0,1,N)
#b_samps = np.arange(0,1,1./N)
#Y = np.zeros([N,4])
#for i in range(0,N):
#    Y[i,:] = fwdMod([.1,b_samps[i],.1],4)


#phi = np.zeros([N,3])
#phi[:,0] = np.ones(N)
#phi[:,1] = b_samps
#phi[:,2] = b_samps**2

#q,r = npla.qr(phi)
#p = np.dot(q.T,Y)
#Z = np.dot(npla.inv(r),p)

#print('Which variable do you want to check?')
#another = True
#while another==True:
#    var = input('Enter variable: ')
#    plt.scatter(phi[:,1],Y[:,var])
#    plt.plot(phi[:,1],np.dot(phi,Z)[:,var])
#    plt.show()
#    another = input('See another plot?\n')
