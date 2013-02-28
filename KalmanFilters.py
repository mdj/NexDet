#!/usr/bin/env python
# encoding: utf-8
"""
KalmanFilters.py



Created by Morten Dam Jørgensen on 2013-02-04.
Copyright (c) 2013 Niels Bohr Institute, Copenhagen. All rights reserved.
"""
from __future__ import division
import sys
import os
import unittest
import numpy as np
import copy
from pprint import pprint
from ROOT import *
DEBUG = False
from math import copysign

np.set_printoptions(edgeitems=3,infstr='Inf',linewidth=400, nanstr='NaN', precision=8, suppress=False, threshold=1000)

class UKF(object):
    """docstring for UKF"""
    def __init__(self):
        super(UKF, self).__init__()
    

    def smooth1(self, M,P,f=None,Q=None,f_param=None, alpha=None, beta=None, kappa=None,mat=0,same_p=True):
        if not f:
            np.eye(M.shape[0])

        if Q is None:
            np.zeros(M.shape[0])


        
          # %
          # % Extend Q if NxN matrix
          # %
          # if size(Q,3)==1
          #   Q = repmat(Q,[1 1 size(M,2)]);
          # end
        # print Q
        if len(Q.shape) < 3 or (len(Q.shape) > 2 and Q.shape[2] ==1):
            # Q= np.tile(Q,(1,1,M.shape[1]))
            Q = np.tile(Q[:,:,np.newaxis],[1,1,M.shape[1]])


          # % Run the smoother
          # %



        D = np.zeros((M.shape[0], M.shape[0], M.shape[1]))

        for k in range(M.shape[1]-2,-1,-1):
            if f_param is None:
                params = None
            elif same_p:
                params = f_param
            else:
                params = f_param[k]

            tr_param = {"alpha":alpha, "beta:" : beta, "kappa": kappa, "mat": mat}
            m_pred, P_pred, C, Xo, Yo, wo  = self.transform(np.array([M[:,k]]).T, P[:,:,k], f, params, tr_param)
            P_pred = P_pred + Q[:,:,k]
            D[:,:,k] = np.linalg.solve(P_pred.T,C.T).T         # D = C / P_pred

            M[:,k] = (np.array([M[:,k]]).T + np.dot(D[:,:,k] , (np.array([M[:,k+1]]).T -m_pred))).T
            P[:,:,k] = P[:,:,k] + np.dot(np.dot(D[:,:,k],(P[:,:,k+1] - P_pred)), D[:,:,k].T)

        return M,P,D
    def gauss_pdf(self,X,M,S):
        '''
        GAUSS_PDF  Multivariate Gaussian PDF
        
         Syntax:
           [P,E] = GAUSS_PDF(X,M,S)
        
         In:
           X - Dx1 value or N values as DxN matrix
           M - Dx1 mean of distibution or N values as DxN matrix.
           S - DxD covariance matrix
        
         Out:
           P - Probability of X. 
           E - Negative logarithm of P
           
         Description:
           Calculate values of PDF (Probability Density
           Function) of multivariate Gaussian distribution
        
            N(X | M, S)
        
           Function returns probability of X in PDF. If multiple
           X's or M's are given (as multiple columns), function
           returns probabilities for each of them. X's and M's are
           repeated to match each other, S must be the same for all.
        
        '''

        if M.shape[1] == 1:
            S = S + 0.1 * np.eye(S.shape[0])
            # print S,np.finfo(float).eps
            DX = X-np.tile(M,(1,X.shape[1]))
            E = 0.5*np.sum(DX * np.linalg.lstsq(S,DX)[0])
            d = M.shape[0]
            E = E + 0.5 * d * np.log(2*np.pi) + 0.5 *  np.log(np.linalg.det(S))

            P = np.exp(-E)

            # From Bishop page 689 equivalent to the above
            # P1 = 1/(2*np.pi)**(d/2) * 1/np.sqrt(np.linalg.det(S)) * np.exp(-0.5 * np.dot(np.dot(DX.T ,np.linalg.inv(S)) , DX))   

            chi2 = np.dot(np.dot(DX.T ,np.linalg.inv(S)) , DX)
            return P,E,chi2
        elif X.shape[1] == 1:
            DX = np.tile(X,(1,M.shape[1])) - M
            E = 0.5 * np.sum(DX * np.linalg.lstsq(S,DX)[0])
            d = M.shape[0]
            E = E + 0.5 * d * np.log(2*np.pi) + 0.5 * np.log(np.linalg.det(S))
            P = np.exp(-E)

        else:
            DX = X-M
            E = 0.5*DX.T*np.linalg.lstsq(S,DX)[0]
            d = M.shape[0]
            E = E + 0.5 * d * np.log(2*np.pi) + 0.5 * np.log(np.linalg.det(S))
            P = np.exp(-E)

        return P,E

    def update1(self, M,P,Y,h,R,h_param=None, alpha=None,beta=None,kappa=None,mat=0):
        '''
            Syntax:
        [M,P,K,MU,S,LH] = UKF_UPDATE1(M,P,Y,h,R,param,alpha,beta,kappa,mat)
        In:
        M  - Mean state estimate after prediction step
        P  - State covariance after prediction step
        Y  - Measurement vector.
        h  - Measurement model function as a matrix H defining
           linear function h(x) = H*x, inline function,
           function handle or name of function in
           form h(x,param)
        R  - Measurement covariance.
        h_param - Parameters of h               (optional, default empty)
        alpha - Transformation parameter      (optional)
        beta  - Transformation parameter      (optional)
        kappa - Transformation parameter      (optional)
        mat   - If 1 uses matrix form         (optional, default 0)
        Out:
        M  - Updated state mean
        P  - Updated state covariance
        K  - Computed Kalman gain
        MU - Predictive mean of Y
        S  - Predictive covariance Y
        LH - Predictive probability (likelihood) of measurement.

        Description:
        Perform additive form Discrete Unscented Kalman Filter (UKF)
        measurement update step. Assumes additive measurement
        noise.
        Function h should be such that it can be given
        DxN matrix of N sigma Dx1 points and it returns 
        the corresponding measurements for each sigma
        point. This function should also make sure that
        the returned sigma points are compatible such that
        there are no 2pi jumps in angles etc.
        Example:
        h = inline('atan2(x(2,:)-s(2),x(1,:)-s(1))','x','s');
        [M2,P2] = ukf_update(M1,P1,Y,h,R,S);
        '''


        tr_param = {"alpha":alpha, "beta:" : beta, "kappa": kappa, "mat": mat}


        MU, S, C, Xo, Yo, wo = self.transform(M,P,h,h_param,tr_param)

        S = S + R
        K = np.linalg.solve(S.T,C.T).T         # K = C / S
        M = M + np.dot(K,(Y - MU))
        P = P - np.dot(K,np.dot(S,K.T))

        LH = self.gauss_pdf(Y,MU,S)

        return M,P,K,MU,S,LH

    def update2(self, M,P,Y,h,R,h_param=None, alpha=None, beta=None, kappa=None,mat=0):

        m = M.shape[0]
        n = R.shape[0]

        MA = np.zeros((m+n,1))
        MA[0:m,:] = M

        PA = np.zeros((P.shape[0]+R.shape[0],P.shape[0]+R.shape[0]))
        PA[0:P.shape[0],0:P.shape[0]] = P
        PA[P.shape[0]:,P.shape[0]:] = R

        tr_param = {"alpha":alpha, "beta:" : beta, "kappa": kappa, "mat": mat}
        MU, S, C, Xo, Yo, wo = self.transform(MA,PA,h,h_param,tr_param)

        K = np.linalg.solve(S.T,C.T).T
        MA = MA + np.dot(K , (Y - MU))
        PA = PA - np.dot(K , np.dot(S , K.T))
        M = MA[0:m,:]
        P = PA[0:m,0:m]

        LH = self.gauss_pdf(Y,MU,S)
        return M,P,K,MU,S,LH


    def update3(self, M,P,Y,h,R,X,w,h_param=None, alpha=None, beta=None, kappa=None,mat=0):

        tr_param = {"alpha":alpha, "beta:" : beta, "kappa": kappa, "mat": mat,"X": X, "w" :w}
        MU, S, C, X, Y_s, wo = self.transform(M,P,h,h_param,tr_param)

        S = S + R
        K = np.linalg.solve(S.T,C.T).T         # K = C / S
        M = M + np.dot(K,(Y - MU))
        P = P - np.dot(K,np.dot(S,K.T))

        LH = self.gauss_pdf(Y,MU,S)

        return M,P,K,MU,S,LH


    def predict1(self, M, P, f = None, Q=None, f_param=None, alpha=None, beta=None, kappa=None, mat=0):
        '''
        UKF_PREDICT1  Nonaugmented (Additive) UKF prediction step
         Syntax:
           [M,P] = UKF_PREDICT1(M,P,f,Q,f_param,alpha,beta,kappa,mat)
         In:
           M - Nx1 mean state estimate of previous step
           P - NxN state covariance of previous step
           f - Dynamic model function as a matrix A defining
               linear function a(x) = A*x, inline function,
               function handle or name of function in
               form a(x,param)                   (optional, default eye())
           Q - Process noise of discrete model   (optional, default zero)
           f_param - Parameters of f               (optional, default empty)
           alpha - Transformation parameter      (optional)
           beta  - Transformation parameter      (optional)
           kappa - Transformation parameter      (optional)
           mat   - If 1 uses matrix form         (optional, default 0)
         Out:
           M - Updated state mean
           P - Updated state covariance
         Description:
           Perform additive form Unscented Kalman Filter prediction step.
           Function a should be such that it can be given
           DxN matrix of N sigma Dx1 points and it returns 
           the corresponding predictions for each sigma
           point. 

        '''
        if f is None:
            f = np.eye(M.shape[0])

        if Q is None:
            Q = np.zeros(M.shape[0])



        tr_param = {"alpha":alpha, "beta:" : beta, "kappa": kappa, "mat": mat}

        M, P, D, X, Y, w = self.transform(M,P,f,f_param,tr_param) # Do transform
        P = P + Q # Add process noise

        return M, P, D

    def predict2(self, M, P, f, Q = None, f_param = None, alpha = None, beta = None, kappa = None, mat=0):
        

    
        # %
        # % Do transform
        # % and add process noise
        # %
        m = M.shape[0]
        n = Q.shape[0]

        MA = np.zeros((m+n, 1))
        MA[0:m,:]= M

        PA = np.zeros((P.shape[0]+Q.shape[0],P.shape[0]+Q.shape[0]))
        PA[:P.shape[0],:P.shape[0]] = P
        PA[P.shape[0]:,P.shape[0]:] = Q
        tr_param = {"alpha":alpha, "beta:" : beta, "kappa": kappa, "mat": mat}
        M, P, D, X, Y, w = self.transform(MA,PA,f,f_param,tr_param)

        return M,P

    def predict3(self, M, P, f, Q, R, f_param = None, alpha = None, beta = None, kappa = None, mat = 0):
        
    
        # %
        # % Do transform
        # % and add process noise
        # %
        m = M.shape[0]
        p = P.shape[0]
        q = Q.shape[0]
        r = R.shape[0]

        MA = np.zeros((m+q+r,1))
        MA[0:m,:] = M

        PA = np.zeros((p+q+r,p+q+r))
        i1 = p
        i2 = i1+q
        PA[:i1,:i1] = P
        PA[i1:i2,i1:i2] = Q
        PA[i2:,i2:] = R


        tr_param = {"alpha":alpha, "beta:" : beta, "kappa": kappa, "mat": mat}
        # M, P, D, X, Y, w = self.transform(MA,PA,f,f_param,tr_param)
        M,P,C,X_s,X_pred,w = self.transform(MA,PA,f,f_param,tr_param)

        # Save sigma points
        X = X_s;
        X[0:X_pred.shape[0],:] = X_pred

        return M,P,X,w,C

    def transform(self, M,P,g,g_param={},tr_param={}):
        '''
        Syntax:
          [mu,S,C,X,Y,w] = UT_TRANSFORM(M,P,g,g_param,tr_param)
        In:
          M - Random variable mean (Nx1 column vector)
          P - Random variable covariance (NxN pos.def. matrix)
          g - Transformation function of the form g(x,param) as
              matrix, inline function, function name or function reference
          g_param - Parameters of g               (optional, default empty)
          tr_param - Parameters of the transformation as:       
              alpha = tr_param{1} - Transformation parameter      (optional)
              beta  = tr_param{2} - Transformation parameter      (optional)
              kappa = tr_param{3} - Transformation parameter      (optional)
              mat   = tr_param{4} - If 1 uses matrix form         (optional, default 0)
              X     = tr_param{5} - Sigma points of x
              w     = tr_param{6} - Weights as cell array {mean-weights,cov-weights,c}
        Out:
          mu - Estimated mean of y
           S - Estimated covariance of y
           C - Estimated cross-covariance of x and y
           X - Sigma points of x
           Y - Sigma points of y
           w - Weights as cell array {mean-weights,cov-weights,c}
        Description:
          ...
          For default values of parameters, see UT_WEIGHTS.

        '''

        alpha = None
        beta = None
        kappa = None
        mat = False
        X = None
        w = None


        if "alpha" in tr_param:
            alpha = tr_param["alpha"]


        if "beta" in tr_param:
            beta = tr_param["beta"]


        if "kappa" in tr_param:
            kappa = tr_param["kappa"]

        if "mat" in tr_param:
            mat = tr_param["mat"]

        if "X" in tr_param:
            X = tr_param["X"]


        if "w" in tr_param:
            w = tr_param["w"]


        # Calculate sigma points

        if w:

            WM = w[0]
            c  = w[2]
            if mat:
                W  = w[1]
            else:
                WC = w[1]
        elif mat:
            WM,W,c = self.mweights(M.shape[0],alpha,beta,kappa)
            X = self.sigmas(M,P,c)
            w = [WM,W,c]
        else:

            WM,WC,c = self.weights(M.shape[0],alpha,beta,kappa)
            X = self.sigmas(M,P,c)
            w = [WM,WC,c]
          
        #
        # Propagate through the function
        #

        if hasattr(g, '__call__'): # a function is to be called...
            Y=np.zeros((X.shape[0], X.shape[1]))
            for i in xrange(X.shape[1]):
                o =  g(X[:,i], g_param)
                if i == 0:
                    Y = np.zeros((len(o), X.shape[1]))

                Y[:,i] = g(X[:,i], g_param)

        else: # OTherwise a simple numeric propagation matrix
            Y = np.dot(g,X)

        if mat:
            mu = np.dot(Y,WM)
            S = np.dot(np.dot(Y,W),Y.T)
            C = np.dot(np.dot(X,W),Y.T)
        else:
            mu = np.zeros((Y.shape[0],1))
            S = np.zeros((Y.shape[0],Y.shape[0]))
            C = np.zeros((M.shape[0],Y.shape[0]))


            for i in xrange(X.shape[1]):
                mu[:,0] = mu[:,0] +  WM[i,0] * Y[:,i]

            for i in xrange(X.shape[1]):
                S = S + WC[i,0] * np.dot((Y[:,i]- mu.T).T, (Y[:,i]- mu.T))
                C = C + WC[i,0] *  np.dot(np.array([X[:,i]]).T -M, (Y[:,i] - mu.T))

        return mu, S, C, X, Y, w



    def mweights(self, n,alpha=None,beta=None,kappa=None):
        '''
        Syntax:
          [WM,W,c] = mweights(n,alpha,beta,kappa)
        In:
          n     - Dimensionality of random variable
          alpha - Transformation parameter  (optional, default 0.5)
          beta  - Transformation parameter  (optional, default 2)
          kappa - Transformation parameter  (optional, default 3-size(X,1))
        Out:
          WM - Weight vector for mean calculation
           W - Weight matrix for covariance calculation
           c - Scaling constant
        Description:
          Computes matrix form unscented transformation weights.


          [WM,WC,c] = ut_weights(n,alpha,beta,kappa);

          W = eye(length(WC)) - repmat(WM,1,length(WM));
          W = W * diag(WC) * W';
        '''

        WM, WC,c = self.weights(n,alpha, beta, kappa)

        W = np.eye(WC.shape[0]) - np.tile(WM, (1, WM.shape[0]))
        W = np.dot(np.dot(W, np.diag(WC[:,0])),W.T)

        return WM, W, c

    def sigmas(self, M,P,c):
        '''
                Syntax:
          X = sigmas(M,P,c);
        In:
          M - Initial state mean (Nx1 column vector)
          P - Initial state covariance
          c - Parameter returned by UT_WEIGHTS
        Out:
          X - Matrix where 2N+1 sigma points are as columns
        Description:
          Generates sigma points and associated weights for Gaussian
          initial distribution N(M,P). For default values of parameters
          alpha, beta and kappa see UT_WEIGHTS.


          Chould be 

                A = chol(P)';
                X = [zeros(size(M)) A -A];
                X = sqrt(c)*X + repmat(M,1,size(X,2));
        '''
        A =  np.linalg.cholesky(P)
        Y = np.tile(M, (1,len(M)))
        X = np.concatenate((M,Y+np.sqrt(c)*A, Y-np.sqrt(c)*A), axis=1)

        return X
    def weights(self, n, alpha=None, beta=None, kappa=None):
        '''
        UT_WEIGHTS - Generate unscented transformation weights
         Syntax:
           [WM,WC,c] = ut_weights(n,alpha,beta,kappa)
         In:
           n     - Dimensionality of random variable
           alpha - Transformation parameter  (optional, default 0.5)
           beta  - Transformation parameter  (optional, default 2)
           kappa - Transformation parameter  (optional, default 3-n)
         Out:
           WM - Weights for mean calculation
           WC - Weights for covariance calculation
            c - Scaling constant
         Description:
           Computes unscented transformation weights.
        '''
        if not alpha:
            alpha = 1
        if not beta:
            beta = 0
        if not kappa:
            kappa = 3 - n

        lambd = alpha*alpha * (n + kappa) - n

        WM = np.zeros([2*n+1,1])
        WC = np.zeros([2*n+1,1])
        for j in xrange(2*n+1):
            if j == 0:
                wm = lambd / (n+lambd)
                wc = lambd / (n+lambd) + (1 - alpha*alpha + beta)
            else:
                wm = 1 / (2*(n+lambd))
                wc = wm

            WM[j,0] = wm
            WC[j,0] = wc

        c = n+lambd

        return WM, WC, c


class UKFTest(unittest.TestCase):
    def SetUp(self):
        pass

    def test_atomic(self):
        ukf = UKF()

        # WM, WC, c =  ukf.weights(3,4,33,2)#, alpha=1e-3, beta=2, kappa=0)
        # print WM
        # print WC
        # print c

        # print 80*"0"
        # WM, WC, c =  ukf.mweights(3,4,33,2)# alpha=1e-3, beta=2, kappa=0)
        # print WM
        # print WC
        # print c


        # M = np.array([[2,3,4]]).T

        # P = np.array([[2,-1,0],
        #               [-1,2,-1],
        #               [0,-1,2]])
        # print M,P
        # # X = ukf.sigmas(M,P,c)
        # # print X

        def g(state, aux): 
            # print "sstate"
            return 0.5 * state

        # g_param = {}
        # tr_param = {}
        # M, P, D= ukf.predict1(M,P,g)
        # print 80*"xx"
        # print M
        # print P
        # print D

        # M = np.array([[2,3,4]]).T

        # P = np.array([[2,-1,0],
        #               [-1,2,-1],
        #               [0,-1,2]])

        # Q = 32.3*np.eye(M.shape[0])
        # M, P= ukf.predict2(M,P,g,Q)
        # print 80*"xx"
        # print M
        # print P
        # print D


        # M = np.array([[2,3,4]]).T

        # P = np.array([[2,-1,0],
        #               [-1,2,-1],
        #               [0,-1,2]])

        # Q = 32.3*np.eye(M.shape[0])
        # R = 3.3*np.eye(M.shape[0]-1)
        # M,P,X,w,C = ukf.predict3(M,P,g,Q,R)
        # print 80*"xx"
        # print "M"
        # print M
        # print "P"
        # print P
        # print "X"
        # print X
        # print "w"
        # print w
        # print "C"
        # print C


        # M = np.array([[2,3,4]]).T

        # P = np.array([[2,-1,0],
        #               [-1,2,-1],
        #               [0,-1,2]])
        # Y = np.array([[2.9,4.3]]).T
        # R = 3.3*np.eye(M.shape[0]-1)
        def h(measurement, aux):
            return [measurement[1], measurement[2]]

        # M,P,K,MU,S,LH = ukf.update2(M,P,Y,h,R)
        # print "M\n",M
        # print "P\n",P
        # print "LH\n",LH


        m0 = np.array([[2,3,4]]).T
        m = copy.deepcopy(m0)

        p0 = np.array([[2,-1,0],
              [-1,2,-1],
              [0,-1,2]])
        P = copy.deepcopy(p0)

        # We miss Y
        Y = np.array([[3,4],[1.5,3],[0.7,1.5]]).T

        Q = 0.3*np.eye(m0.shape[0])
        R = 3.3*np.eye(Y.shape[0])


        MM = np.zeros((m.shape[0], Y.shape[1]))
        PP = np.zeros((m.shape[0], m.shape[0], Y.shape[1]))


        for k in range(Y.shape[1]):
            m,P,D = ukf.predict1(m,P,g,Q)
            m,P,K,MU,S,LH = ukf.update1(m,P,np.array([Y[:,k]]).T,h,R)
            print "LH", LH
            MM[:,k] = m.T
            PP[:,:,k] = P


        SM,SP,D = ukf.smooth1(MM,PP,g,Q)
        print "SM\n",SM
        print "SP\n",SP

        for i in xrange(SP.shape[2]):
            print SP[:,:,i]




class UncentedKalmanFilter(object):
    """
    Implementation based on http://www.mathworks.com/matlabcentral/fileexchange/18217-learning-the-unscented-kalman-filter
    
    UKF   Unscented Kalman Filter for nonlinear dynamic systems
    [x, P] = UncentedKalmanFilter.step(f,x,P,h,z,Q,R) returns state estimate, x and state covariance, P 
    for nonlinear dynamic system (for simplicity, noises are assumed as additive):
              x_k+1 = f(x_k) + w_k
              z_k   = h(x_k) + v_k
    where w ~ N(0,Q) meaning w is gaussian noise with covariance Q
          v ~ N(0,R) meaning v is gaussian noise with covariance R
    Inputs:   f: function handle for f(x)
              x: "a priori" state estimate
              P: "a priori" estimated state covariance
              h: fanction handle for h(x)
              z: current measurement
              Q: process noise covariance 
              R: measurement noise covariance
    Output:   x: "a posteriori" state estimate
              P: "a posteriori" state covariance
    
    """
    def __init__(self):
        super(UncentedKalmanFilter, self).__init__()
        
        self.alpha=1e-3;                                 #default, tunable
        self.ki=0;                                       #default, tunable
        self.beta=2;                                     #default, tunable


    def step(self, fstate, x, P, hmeas, z, Q, R, aux = None):
        """
        Calculate
        """

        L=x.shape[0];                                 #numer of states
        m=z.shape[0];                                 #numer of measurements
        
        lam=self.alpha**2 * (L+self.ki) - L;                    #scaling factor
        c=L+lam;                                         #scaling factor

        self.Wm = np.zeros([1,2*L+1])
        self.Wm += 0.5/c
        self.Wm[0,0] = lam/c

        self.Wc=copy.deepcopy(self.Wm);
        self.Wc[0,0]=self.Wc[0,0]+(1.0-self.alpha**2+self.beta);               #weights for covariance
        
        
        c=np.sqrt(c);
        # print c
        self.X=self.sigmas(x,P,c);                            #sigma points around x
        
        
        # Prediction 
        [self.x1,self.X1,self.P1,self.X2]=self.ut(fstate,self.X,self.Wm,self.Wc,L,Q, aux);          #unscented transformation of process


        # Correction
        ## X1=sigmas(x1,P1,c);                         #sigma points around x1
        ## X2=X1-x1(:,ones(1,size(X1,2)));             #deviation of X1
        [self.z1,self.Z1,self.P2,self.Z2]=self.ut(hmeas,self.X1,self.Wm,self.Wc, m, R, aux);       #unscented transformation of measurments


        self.P12=np.dot(np.dot(self.X2,np.diag(self.Wc[0])),self.Z2.T)                        #transformed cross-covariance
        self.K=np.dot(self.P12,np.linalg.inv(self.P2))

        self.x=self.x1+np.dot(self.K,(z-self.z1))                              #state update
        self.P=self.P1-np.dot(self.K,self.P12.T)                                #covariance update


        # print R-self.P2
        self.pull = (z-self.z1) / np.sqrt(R - self.P2)

        # print R.shape, self.K.shape, self.P.shape
        Rkk = R - np.dot(self.K.T,np.dot(self.P, self.K))
        r = (z-self.z1);

        self.chi2 = np.dot(np.dot(r, np.linalg.inv(Rkk)), r.T)
        
        return self.x, self.P

    def ut(self, f, X, Wm, Wc, n, R, aux):
        """
        Unscented Transformation
        
        Input:
                f: nonlinear map
                X: sigma points
               Wm: weights for mean
               Wc: weights for covraiance
                n: numer of outputs of f
                R: additive covariance
        Output:
                y: transformed mean
                Y: transformed smapling points
                P: transformed covariance
               Y1: transformed deviations
        """

        if DEBUG:
            print "in ut(f, X, Wm, Wc, n, R, aux)"
            
        L=X.shape[1]
        y=np.zeros([n,1])
        Y=np.zeros([n,L])
        
        wcnorm  =np.zeros([n,1])

        for k in xrange(L):
            Y[:,k]= np.array([f(X[:,k], aux)])
            y =  y +  np.array([Wm[:,k][0] * Y[:,k]]).T

        Y1 = Y - np.tile(y, (1,L)) # center around mean prediction
        P = np.dot(np.dot(Y1,np.diag(Wc[0])), Y1.T) + R # calculate covariance
        
        return y, Y, P, Y1
    

    def sigmas(self, x, P, c):
        """
        Sigma points around reference point
        Inputs:
               x: reference point
               P: covariance
               c: coefficient
        Output:
              X: Sigma points
        """
        try:
            A = c * np.linalg.cholesky(P).T
            Y = np.tile(x, (1,len(x)))
            X =  np.concatenate((x,Y+A, Y-A), axis=1)
        except:
            print 80*"+"
            print "Cholesky decomposition failed, showing the covariance matrix"
            print P
            print 80*"+"
            raw_input("End program [enter] ")
            sys.exit(1)
        return X
        
        
    def best_fit(self, event, fstate, hmeas, gEve = None):
        """same as self.step() but for vectors of measurement vectors, calculating the chi^2 stuff and finding the best"""
        

        from PhysicsObjects import RecoTrack


        # initial tracklets:
        reco_trks = []

        di = event.detector_fz[0]
        det0 = event.detectors[di]
        i = 0
        
        def add_to_param(trk, param, z):
            """docstring for add_to_param"""
            for k,p in enumerate(param):
                trk.param_graph[k].SetPoint(trk.param_graph[k].GetN(), z, p[0])
                
        for hit in det0.hits:
            i += 1
            print 80*"=="
            # print "Hit %d in First layer encountered, straight line approx at qoverp = 5 GeV" 
            
            pos = np.array(hit.position[:3])        
            mom = 5.0
            q = 1.0
            qoverp = q/mom
            
            e = ((pos-event.ip0) / np.linalg.norm(pos-event.ip0))
            dx = e[0]
            dy = e[1]
            dz = e[2]
            tx = dx/dz
            ty = dy/dz
            r = np.array([[pos[0],pos[1],tx,ty,qoverp]]).T
            

            # DEBUG CHEAT USING TRUE MOMENTA
            # r = np.array([[pos[0],pos[1],tx,ty,copysign(qoverp, hit.true_track.r[4])]]).T
            
            P = np.array([  [1.0, 0, 0, 0, 0],
                            [0, 1.0, 0, 0, 0],
                            [0, 0, 1.0, 0, 0.0],
                            [0, 0, 0, 1.0, 0.0],
                            [0, 0, 0.0,0.0, 1.0]])

            P = 1.0e-3 * np.eye(len(r))
            Q = 1e-6 * np.eye(len(r)) # Covariance of process
            
            trklet_associated_hits =  TEvePointSet("Associated hit trk %d" % i)
            trklet_associated_hits.SetMarkerColor(kAzure)
            trklet_associated_hits.SetMarkerSize(1.8)
            trklet_associated_hits.SetMarkerStyle(4);

            trklet_extrapolated =  TEvePointSet("Extrapolated  trk %d" % i)
            trklet_extrapolated.SetMarkerColor(kWhite)
            trklet_extrapolated.SetMarkerSize(1.8)
            trklet_extrapolated.SetMarkerStyle(4);


            trklet_sigmapoints =  TEvePointSet("Sigmapoints in trk %d" % i)
            trklet_sigmapoints.SetMarkerColor(kYellow)
            trklet_sigmapoints.SetMarkerSize(1.0)
            trklet_sigmapoints.SetMarkerStyle(4)

            trklet_sigmapoints_out =  TEvePointSet("Sigmapoints out trk %d" % i)
            trklet_sigmapoints_out.SetMarkerColor(kGreen)
            trklet_sigmapoints_out.SetMarkerSize(1.0)
            trklet_sigmapoints_out.SetMarkerStyle(4)


            trklet_associated_hits.SetNextPoint(r[0], r[1], det0.z)
            trklet_extrapolated.SetNextPoint(r[0], r[1], det0.z)

            tracklet = RecoTrack()
            tracklet.g_hits = trklet_associated_hits
            tracklet.g_extr = trklet_extrapolated
            tracklet.g_extr_sig = trklet_sigmapoints
            tracklet.g_extr_sig_out = trklet_sigmapoints_out
            
            tracklet.param_graph = [TGraph(), TGraph(), TGraph(), TGraph(), TGraph()]
            tracklet.r = r
            tracklet.z = det0.z
            tracklet.P = P
            tracklet.Q = Q
            
            tracklet.r_at_surface = [] # These three _at_surface lists contains the results at each detector.
            tracklet.P_at_surface = [] # They can be used for smoothing.
            tracklet.z_at_surface = []
            tracklet.measurement_at_surface = []
            
            tracklet.hits_on_track.append(hit)
            reco_trks.append(tracklet)

            
            add_to_param(tracklet, r,det0.z) # add to evolution graph
            
        # Loop over detector planes
        for det_idx in xrange(1,4):
            det = event.detectors[event.detector_fz[det_idx]] # Next layer
            print "Tracking at Plane %s, z=%2.2e m" % (det.name, det.z)
            hits = copy.copy(det.hits) # Shallow copy the hits container
            for trk in reco_trks:

                L=trk.r.shape[0];                                 #numer of states
                lam=self.alpha**2 * (L+self.ki) - L;                    #scaling factor
                c=L+lam;                                         #scaling factor

                Wm = np.zeros([1,2*L+1])
                Wm += 0.5/c
                Wm[0,0] = lam/c
                Wc=copy.deepcopy(Wm);
                Wc[0,0]=Wc[0,0]+(1.0-self.alpha**2+self.beta);               #weights for covariance                
                c=np.sqrt(c);
                X=self.sigmas(trk.r,trk.P,c);                            #sigma points around x
                
                # Prediction 
                [x1,X1,P1,X2]=self.ut(fstate,X,Wm,Wc,L,trk.Q, {"z" : trk.z, "zf" : det.z});          #unscented transformation of process

                hitset = []
                for meas in hits:  # Compare with all hits
                    z = np.array([[meas.position[0], meas.position[1]]]).T
                    m=z.shape[0];                                 #numer of measurements
                    R = 1.0e-7 * np.eye(len(z)) # Create a noise matrix for each measurement

                    # # Prediction of measurement
                    # z1 == predicted measurement
                    # P2 == predicted measurement covariance.
                    [z1,Z1,P2,Z2]=self.ut(hmeas,X1,Wm,Wc, m, R, {"z" : trk.z, "zf" : det.z});       #unscented transformation of measurments

                    for sp in xrange(2*L+1): trk.g_extr_sig_out.SetNextPoint(Z1[0,sp], Z1[1,sp], det.z)

                                            
                    P12=np.dot(np.dot(X2,np.diag(Wc[0])),Z2.T)    #The state-measurement cross-covariance matrix                    
                    K=np.dot(P12,np.linalg.inv(P2)) # Kalman gain
                    
                    
                    xguess = x1 + np.dot(K,(z-z1))                              #state update
                    Pguess = P1 - np.dot(K,P12.T)                                #covariance update
                    
                    # Attempt at chi2 stuff
                    Rkk = R + np.dot(K.T,np.dot(Pguess, K))
                    # Rkk = R - np.dot(np.dot(P12.T, Pguess), P12)
                    r = (z-z1)
                    chi2 = np.dot(np.dot(r.T, np.linalg.inv(Rkk)), r)[0,0]
                    # print np.linalg.norm(r), chi2, trk.hits_on_track[0].true_particle is meas.true_particle
                    hitset.append([meas, chi2, xguess, Pguess, trk.hits_on_track[0].true_particle is meas.true_particle, r])

                # pprint(hitset)
                # print 80*"-"
                minchi = 999999999
                hitsel = None
                for i,hit in enumerate(hitset):
                    if hit[1] < minchi:
                        minchi = hit[1]
                        hitsel = hit

                    # print "min(chi2) = %2.20f" % minchi

                if hitsel:
                    print "found true hit? ", hitsel[4], " residual: ", hitsel[5].T
                    trk.hits_on_track.append(hitsel[0])
                    trk.r = hitsel[2]
                    trk.P = hitsel[3]
                    trk.z = det.z
                    
                    trk.r_at_surface.append(trk.r)
                    trk.P_at_surface.append(trk.P)
                    trk.z_at_surface.append(det.z)
                    trk.measurement_at_surface.append(hitsel[0])
                    
                    trk.g_hits.SetNextPoint(hitsel[0].position[0], hitsel[0].position[1], hitsel[0].position[2])
                    trk.g_extr.SetNextPoint(trk.r[0], trk.r[1], trk.z)

                    print "Estimated track ", trk.r.T
                    print "True parameters ",  hitsel[0].true_parameters_at_surface[0]
                
                    add_to_param(trk, trk.r,det.z) # add to evolution graph
                    hits.remove(hitsel[0]) # remove the hit from hte container

        
        for trk in reco_trks:
            gEve.AddElement(trk.g_hits)
            gEve.AddElement(trk.g_extr)
            gEve.AddElement(trk.g_extr_sig)
            gEve.AddElement(trk.g_extr_sig_out )
        
        
            gEve.Redraw3D()
            
        parameter_evolution_canvas = TCanvas("parameter_evolution_canvas")
        parameter_evolution_canvas.Divide(1,5)

        for i,trk in enumerate(reco_trks):
            for j,p in enumerate(trk.param_graph):
                # trk.param_graph[i].SetPoint(param_graph[i].GetN(), z, p[0])
                parameter_evolution_canvas.cd(1+j)
                p.SetLineColor(i+1)

                if i == 0:
                    p.Draw("ALP")
                else:
                    p.Draw("LP SAME")
        
        
        raw_input("done in kalman")
        
# class UncentedKalmanFilterTest(unittest.TestCase):
#     def setUp(self):
#         pass
        
#     def test_intialization(self):
#         """docstring for test_intialization"""
        


#         # clc;
#         n=3      #number of state
#         q=0.1    #std of process 
#         r=0.1    #std of measurement
#         Q=q**2* np.eye(n) # covariance of process
#         self.assertEqual(Q.shape, (3,3))

#         R=np.array([[r**2]])        #covariance of measurement  

#         def f(x, aux=None): return np.array([x[1],x[2],0.05*x[0]*(x[1]+x[2])]) # nonlinear state equations
#         def h(x, aux=None): return np.array([x[0]]) # Measurement Function
        
#         s = np.array([[0,0,1]]).T
#         self.assertEqual(s.shape, (3,1))
        
#         x = s + q * np.random.randn(3,1)
#         self.assertEqual(x.shape, (3,1))

#         P = np.eye(n)
#         self.assertEqual(P.shape, (3,3))
        
#         N=500
#         xV = np.zeros([n,N])
#         sV = np.zeros([n,N])
#         # print sV.shape
#         zV = np.zeros([N,1])

#         self.assertEqual(xV.shape, (n,N))
#         self.assertEqual(sV.shape, (n,N))
#         self.assertEqual(zV.shape, (N,1))
        
#         # # Create a new filter
#         kalman = UncentedKalmanFilter()
#         # kalman.initialize()
    
#         pulls = [0]
#         for k in xrange(1,N):
#             z = h(s) + r * np.random.randn()
#             sV[:,k] = s.T[0]
#             zV[k] = z
#             x, P = kalman.step(f, x, P, h, z, Q, R) #step(z, R, f, h)
#             # print "x", x
#             # print "z", z
#             xV[:,k] = x.T[0]
#             s = f(s) + q*np.random.randn()
#             print kalman.chi2
#             pulls.append(kalman.chi2)
            
#         try: # If we have ROOT install try plotting with it
#             from ROOT import TCanvas, TGraph, kBlue, kGreen,TH1F
    
#             xVx = TGraph()
#             xVy = TGraph()
#             xVz = TGraph()
#             xVx.SetLineColor(kGreen)
#             xVy.SetLineColor(kGreen)
#             xVz.SetLineColor(kGreen)

#             sVx = TGraph()
#             sVy = TGraph()
#             sVz = TGraph()
#             sVx.SetLineColor(kBlue)
#             sVy.SetLineColor(kBlue)
#             sVz.SetLineColor(kBlue)
    
#             zVx = TGraph()
#             # zVx.SetLineColor(kBlue)
#             zVx.SetLineStyle(3)
        
#             pullgraf = TH1F("pullgraf","chi graph", 100, 0, 100)
#             for i,j in enumerate(xV[0]):
#                 zVx.SetPoint(i+1, i+1, zV[i])
            
#                 xVx.SetPoint(i+1, i+1, xV[0][i])
#                 xVy.SetPoint(i+1, i+1, xV[1][i])
#                 xVz.SetPoint(i+1, i+1, xV[2][i])
    
#                 sVx.SetPoint(i+1, i+1, sV[0][i])
#                 sVy.SetPoint(i+1, i+1, sV[1][i])
#                 sVz.SetPoint(i+1, i+1, sV[2][i])

#                 pullgraf.Fill(pulls[i])
            
#             c = TCanvas("unittest")
#             c.Divide(1,3)
    
#             c.cd(1)
#             sVx.Draw("ALP")
#             xVx.Draw("LP")
#             zVx.Draw("LP")
        
#             c.cd(2)
#             sVy.Draw("ALP")
#             xVy.Draw("LP")
    
#             c.cd(3)
#             sVz.Draw("ALP")
#             xVz.Draw("LP")
        
#             cc = TCanvas("cc")    
#             pullgraf.Draw("")
        
#             raw_input("Done..")
#         except:
#             pass        
        
#         return self.assertTrue(True)

        
        
        
        
if __name__ == '__main__':
	unittest.main()


#
        # det1 = event.detectors[event.detector_fz[1]] # Next layer
        # hits = copy.copy(det1.hits) # Shallow copy the hits container
        # for trk in reco_trks:
        #     # print trk
        #     # print det1.z

        #     # Calculate prediction on surface
        #     L=trk.r.shape[0];                                 #numer of states

        #     # Calculate Sigma
        #     lam=self.alpha**2*(L+self.ki)-L;                    #scaling factor
        #     c=L+lam;                                 #scaling factor
        #     Wm =   0.5/c+np.zeros([1,2*L+1])
        #     Wm[0,0] = lam/c
        #     Wc=Wm;
        #     Wc[0]=Wc[0]+(1-self.alpha**2+self.beta);               #weights for covariance
        #     c=np.sqrt(c);
        #     X=self.sigmas(trk.r,trk.P,c);                            #sigma points around x
            
        #     aux = {"z" : trk.z, "zf" : det1.z}

        #     # Prediction 
        #     [x1,X1,P1,X2]=self.ut(fstate,X,Wm,Wc,L,trk.Q, aux);          #unscented transformation of process
        #     # print x1


        #     print 80*"-"
        #     print 80*"-"

        #     hitset = []

        #     # Compare with all hits
        #     for meas in hits:
        #         print "Hits left to search in %d" % len(hits)
        #         print "Unscented Kalman Filter from z=%f to %f" % (det0.z, det1.z)

        #         z = np.array([[meas.position[0], meas.position[1]]]).T
        #         m=z.shape[0];                                 #numer of measurements
        #         R = 0.01 * np.eye(len(z)) # Create a noise matrix for each measurement



        #         # # Correction
        #         [z1,Z1,P2,Z2]=self.ut(hmeas,X1,Wm,Wc, m, R, aux);       #unscented transformation of measurments

        #         # z1 == predicted measurement
        #         # P2 == predicted measurement covariance.

        #         P12=np.dot(np.dot(X2,np.diag(Wc[0])),Z2.T)    #The state-measurement cross-covariance matrix

        #         K=np.dot(P12,np.linalg.inv(P2)) # Kalman gain

        #         xguess=x1+np.dot(K,(z-z1))                              #state update
        #         Pguess=P1-np.dot(K,P12.T)                                #covariance update
        #         Rkk = R + np.dot(K.T,np.dot(Pguess, K))

        #         # Rkk = R - np.dot(np.dot(P12.T, Pguess), P12)
        #         r = (z-z1)
        #         chi2 = np.dot(np.dot(r.T, np.linalg.inv(Rkk)), r)

        #         # print np.linalg.norm(r), chi2, trk.hits_on_track[0].true_particle is meas.true_particle
        #         hitset.append([meas, chi2[0,0], xguess, Pguess, trk.hits_on_track[0].true_particle is meas.true_particle])

        #     # pprint(hitset)
        #     print 80*"-"
        #     minchi = 999999999
        #     hitsel = None
        #     for i,hit in enumerate(hitset):
        #         if hit[1] < minchi:
        #             minchi = hit[1]
        #             hitsel = hit
        #         print "min(chi2) = %2.20f" % minchi

        #     print "found true hit? ", hitsel[4]
        #     trk.hits_on_track.append(hitsel[0])
        #     hits.remove(hitsel[0]) # remove the hit from hte container
        #     trk.r = hitsel[2]
        #     trk.P = hitsel[3]
        #     trk.z = det1.z
        # ################## det 2