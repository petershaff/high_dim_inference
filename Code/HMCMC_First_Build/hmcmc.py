execfile('interp.py')

class lillypad_mcmc:
    def __init__(self, post, S, approx='tps', M = [], kwargs):
        self.post = post
        self.d = post.d

        self.S = S
        self.fS = np.array([ self.post(s) for s in S ])

        self.kwargs = kwargs
        self.approx_flag = approx
        
        if self.approx_flag == 'tps':
            self.approx = thin_plate( self.S, self.fS, **self.kwargs )
        
        elif self.approx_flag == 'lqr':
            self.approx = local_quad_reg( self.S, self.fS)
        
        else:
            raise ValueError("I don't recognize that approximation algorithm, try again fool!")

        if M == []:
            self.M = np.eye(self.d)

        else:
            self.M = M
        
        self.curr = self.S[ np.random.choice( range(0, self.S.shape[1]) ) ]
        self.samps = [self.curr]

        self.pot = lambda x: -1*np.log( self.approx.evaluate(x) ) 
        self.steps = np.random.choice(range(10,101))

        
    def propose(self, pot = self.pot, h = 1e-3, q0 = self.curr):
        p0 = np.random.multivariate_normal( np.zeros( self.d ), M )
        x0 = np.concatenate([ q0,p0 ])
        
        path = [x0]
        for i in range(0, steps):
            xt = lpfrg_pshfwd( path[i], h, pot, self.M)
            path.append(xt)

        x0 = -1*x0[(len(xt)/2):len(xt)]
        return([ x0, path ])

    def accept_prob(self, curr, cand, pot = self.pot):
        q0 = curr[ 0:(len(curr)/2)]
        q0_cand = cand[ 0:(len(cand)/2)]

        p0 = curr[ (len(curr)/2):len(curr) ]
        p0_cand = cand[ (len(cand)/2):len(cand) ]

        a = np.min( 1, np.exp( pot(q0) - pot(q0_cand) + np.dot(p0, np.dot(self.M, p0)) + np.dot(p0_cand, np.dot(self.M, p0_cand)) ) )
        return( a )
            
    def refine_test(self, path, tol = 1e-3):
        knn = skn.NearestNeighbors(n_neighbors = int(2*self.d*len(path)) )
        knn.fit(self.S)
        [dists, inds] = knn.kneighbors( np.array(path) ) 

        B = self.S[inds]
        fB = self.fS[inds]

        curr = path[0]
        
        N = B.shape[0]
        a_list = []
        for i in range(0,10):
            sub_inds = np.random.choice( range(0, N), int(.9*N) )
            B_sub = B[sub_inds]
            fB_sub = fB[sub_inds]
            
            if self.approx_flag == 'tps':
                self.approx = thin_plate( B_sub, fB_sub, **self.kwargs )
        
            elif self.approx_flag == 'lqr':
                self.approx = local_quad_reg( B_sub, fB_sub)
        
            else:
                raise ValueError("I don't recognize that approximation algorithm, try again fool!")

            pot = lambda x: -1*np.log( self.approx.evaluate(x) )
            [cand, path]self.propose(pot = pot)

            a_list.append( self.accept_prob(curr, cand, pot = pot) )

        if np.abs( max(a_list) - min(a_list) ) <= tol:
            return(1)

        else:
            return(0)
