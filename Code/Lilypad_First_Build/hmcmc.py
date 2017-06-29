execfile('interp.py')

class lilypad_mcmc:
    def __init__(self, post, S, kwargs, approx='tps', M = None):
        self.post = post
        self.d = S[0].shape[0]

        self.S = S
        self.fS = np.array([ self.post(s) for s in S ])

        self.kwargs = kwargs
        self.approx_flag = approx
        
        if self.approx_flag == 'tps':
            self.approx = thin_plateV2( self.S, self.fS, **self.kwargs )
        
        elif self.approx_flag == 'lqr':
            self.approx = local_quad_reg( self.S, self.fS)
        
        else:
            raise ValueError("I don't recognize that approximation algorithm, try again fool!")

        if M == None:
            self.M = np.eye(self.d)

        else:
            self.M = M
        
        self.curr = self.S[ np.random.choice( range(0, self.S.shape[1]) ) ] + np.random.multivariate_normal(np.zeros(self.d), np.eye(self.d) )
        self.samps = [self.curr]

        self.pot = lambda x: -1*np.log( np.abs( self.approx.evaluate(x) ) )
        self.steps = np.random.choice(range(10,101))

    #Basic proposal function
    def propose(self, h = 1e-3, steps = 10, pot = None, q0 = None):
        if pot == None:
            pot = self.pot
            
        else:
            pot = pot
            
        if q0 == None:
            q0 = self.curr
            
        else:
            q0 = q0
        
        p0 = np.random.multivariate_normal( np.zeros( self.d ), self.M )
        x0 = np.concatenate([ q0,p0 ])
        
        [xt, path] = self.lpfrg(x0, steps, pot, h)
        return([xt, path])
        
    #The Hamiltonian Pushforward as integrated by leapfrog
    def lpfrg(self, x0, steps, pot, h):
        path = [x0]
        for i in range(0, steps):
          xt = lpfrg_pshfwd( path[i], h, pot, self.M)
 
          path.append(xt)

        xt = np.concatenate([ xt[0:(len(xt)/2)], -1*xt[(len(xt)/2):len(xt)] ])
        return([ xt, path ])

    #Calculate probability of accepting candidate
    def accept_prob(self, curr, cand, pot = None):
        if pot == None:
            pot = self.pot
            
        else:
            pot = pot
                    
        q0 = curr[ 0:self.d]
        q0_cand = cand[ 0:self.d]

        p0 = curr[ self.d:(2*self.d) ]
        p0_cand = cand[ self.d:(2*self.d) ]
        
        a = min( 1, np.exp( pot(q0) - pot(q0_cand) + np.dot(p0, np.dot(self.M, p0)) + np.dot(p0_cand, np.dot(self.M, p0_cand)) ) )
        return( a )
    
    #10-fold CV on points adjacent to path from self.propose() to see if it varies acceptance prob enough to add refinement
    def refine_test(self, cand, path, tol = 1e-3, h = 1e-3):
        knn = skn.NearestNeighbors(n_neighbors = int(2*self.d*len(path)) )
        knn.fit(self.S)
        self.knn = knn
        knn.kneighbors( np.array(path)[:,0:2] )
        [dists, inds] = knn.kneighbors( np.array(path)[:,0:2] )
        inds = np.unique(inds)
        self.inds = inds
    
        B = self.S[inds]
        fB = self.fS[inds]

        curr = path[0]
        
        N = B.shape[0]
        a_list = [ self.accept_prob(curr, cand)]
        for i in range(0,10):
            sub_inds = np.random.choice( range(0, N), int(.9*N), replace=False)
            B_sub = B[sub_inds]
            fB_sub = fB[sub_inds]
            
            if self.approx_flag == 'tps':
                approx = thin_plateV2( B_sub, fB_sub, **self.kwargs )
        
            elif self.approx_flag == 'lqr':
                approx = local_quad_reg( B_sub, fB_sub)
        
            else:
                raise ValueError("I don't recognize that approximation algorithm, try again fool!")

            pot = lambda x: -1*np.log( approx.evaluate(x) )
            print( approx.timer)
            [cand, path] = self.lpfrg(curr, steps, pot, h)

            a_list.append( self.accept_prob(curr, cand, pot = pot) )

        if np.abs( max(a_list) - min(a_list) ) <= tol:
            return(1)

        else:
            return(0)

steps = 10

#run_time = time.time()
test = lilypad_mcmc(test_dense.evaluate, S, {})
#[cand, path] = test.propose(10.)
#refine = test.refine_test(cand,path)
#test.accept_prob(path[0],path[1])
#run_time = run_time - time.time()

x_test = np.array([1.5,2.5])
fx_test = test_dense.evaluate(x_test)
test.approx.rank_one_add( x_test, fx_test)
