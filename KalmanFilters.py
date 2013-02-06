#!/usr/bin/env python
# encoding: utf-8
"""
KalmanFilters.py



Created by Morten Dam Jørgensen on 2013-02-04.
Copyright (c) 2013 Niels Bohr Institute, Copenhagen. All rights reserved.
"""

import sys
import os
import unittest
import numpy as np
        

class UncentedKalmanFilter(object):
    """
    Implementation based on http://www.mathworks.com/matlabcentral/fileexchange/18217-learning-the-unscented-kalman-filter
    
    """
    def __init__(self):
        super(UncentedKalmanFilter, self).__init__()


    def step(self, fstate, x, P, hmeas, z, Q, R, aux = None):
        """Everything in one go"""

        L=x.shape[0];                                 #numer of states
        m=z.shape[0];                                 #numer of measurements

        alpha=1e-3;                                 #default, tunable
        ki=0;                                       #default, tunable
        beta=2;                                     #default, tunable
        lam=alpha**2*(L+ki)-L;                    #scaling factor
        c=L+lam;                                 #scaling factor
        Wm =   0.5/c+np.zeros([1,2*L+1])
        Wm[0,0] = lam/c
        # print Wm
        # Wm= [lam/c, 0.5/c+np.zeros([1,2*L])];           #weights for means
        Wc=Wm;
        Wc[0]=Wc[0]+(1-alpha**2+beta);               #weights for covariance
        c=np.sqrt(c);
        # print c
        X=self.sigmas(x,P,c);                            #sigma points around x
        [x1,X1,P1,X2]=self.ut(fstate,X,Wm,Wc,L,Q, aux);          #unscented transformation of process

        ## X1=sigmas(x1,P1,c);                         #sigma points around x1
        ## X2=X1-x1(:,ones(1,size(X1,2)));             #deviation of X1
        [z1,Z1,P2,Z2]=self.ut(hmeas,X1,Wm,Wc,m,R, aux);       #unscented transformation of measurments


        P12=np.dot(np.dot(X2,np.diag(Wc[0])),Z2.T)                        #transformed cross-covariance
        K=np.dot(P12,np.linalg.inv(P2))

        x=x1+np.dot(K,(z-z1))                              #state update
        P=P1-np.dot(K,P12.T)                                #covariance update

        # r = K*(z-z1);
        # chi2 = (r'*inv(P)*r)
        return x, P

    def ut(self, f, X, Wm, Wc, n, R, aux):
        """
        %Input:
        %        f: nonlinear map
        %        X: sigma points
        %       Wm: weights for mean
        %       Wc: weights for covraiance
        %        n: numer of outputs of f
        %        R: additive covariance
        %Output:
        %        y: transformed mean
        %        Y: transformed smapling points
        %        P: transformed covariance
        %       Y1: transformed deviations

        L=size(X,2);
        y=zeros(n,1);
        Y=zeros(n,L);
        for k=1:L            
            Y(:,k)=f(X(:,k));       
            y=y+Wm(k)*Y(:,k);       
        end
        Y1=Y-y(:,ones(1,L));
        P=Y1*diag(Wc)*Y1'+R;%Unscented Transformation
        %Input:
        %        f: nonlinear map
        %        X: sigma points
        %       Wm: weights for mean
        %       Wc: weights for covraiance
        %        n: numer of outputs of f
        %        R: additive covariance
        %Output:
        %        y: transformed mean
        %        Y: transformed smapling points
        %        P: transformed covariance
        %       Y1: transformed deviations

        L=size(X,2);
        y=zeros(n,1);
        Y=zeros(n,L);
        for k=1:L            
            Y(:,k)=f(X(:,k));       
            y=y+Wm(k)*Y(:,k);       
        end
        Y1=Y-y(:,ones(1,L));
        P=Y1*diag(Wc)*Y1'+R;"""

        L=X.shape[1]
        y=np.zeros([n,1])
        Y=np.zeros([n,L])
        # print y
        for k in xrange(L):
            Y[:,k]= np.array([f(X[:,k], aux)])
            y =  y +  np.array([Wm[:,k][0] * Y[:,k]]).T
            
        Y1 = Y - np.tile(y, (1,L))
        P = np.dot(np.dot(Y1,np.diag(Wc[0])), Y1.T) + R
        # print y,Y,P,Y1
        return y, Y, P, Y1
    

    def sigmas(self, x, P, c):
        
        """
                sigmas(x,P,c)
                %Sigma points around reference point
                %Inputs:
                %       x: reference point
                %       P: covariance
                %       c: coefficient
                %Output:
                %       X: Sigma points
                A = c*chol(P)';
                Y = x(:,ones(1,numel(x)));
                X = [x Y+A Y-A];
        """
    
        A = c * np.linalg.cholesky(P).T
        Y = np.tile(x, (1,len(x)))
        X =  np.concatenate((x,Y+A, Y-A), axis=1)

        return X
        
        
        
class UncentedKalmanFilterTest(unittest.TestCase):
    def setUp(self):
        ip0 =  np.array([0, 0, -1.0/2.0 - 0.1])
        hit_x = np.array([-0.00016567089886896337, -0.00039565159911799698, -0.0001225])
        p = 10.0
        p0 = p * ((hit_x-ip0) / np.linalg.norm(hit_x-ip0))

        self.x = np.array([hit_x[0],hit_x[1],hit_x[2],p0[0],p0[1],p0[2], -1])
        self.P = np.array([  [1.0, 0, 0, 0, 0, 0, 0],
                        [0.0, 1, 0, 0, 0, 0, 0],
                        [0.0, 0, 1, 0, 0, 0, 0],
                        [0.0, 0, 0, 1, 0, 0, 0],
                        [0.0, 0, 0, 0, 1, 0, 0],
                        [0.0, 0, 0, 0, 0, 1, 0],
                        [0.0, 0, 0, 0, 0, 0, 1]])    

        self.Q = 0.1**2 * np.eye(len(self.x)) # Covariance of process
        
    def test_intialization(self):
        """docstring for test_intialization"""
        


        # clc;
        n=3      #number of state
        q=0.1    #std of process 
        r=0.1    #std of measurement
        Q=q**2* np.eye(n) # covariance of process
        self.assertEqual(Q.shape, (3,3))

        R=r**2        #covariance of measurement  

        def f(x, aux=None): return np.array([x[1],x[2],0.05*x[0]*(x[1]+x[2])]) # nonlinear state equations
        def h(x, aux=None): return np.array([x[0]]) # Measurement Function
        
        s = np.array([[0,0,1]]).T
        self.assertEqual(s.shape, (3,1))
        
        x = s + q * np.random.randn(3,1)
        self.assertEqual(x.shape, (3,1))

        P = np.eye(n)
        self.assertEqual(P.shape, (3,3))
        
        N=50
        xV = np.zeros([n,N])
        sV = np.zeros([n,N])
        # print sV.shape
        zV = np.zeros([N,1])

        self.assertEqual(xV.shape, (n,N))
        self.assertEqual(sV.shape, (n,N))
        self.assertEqual(zV.shape, (N,1))
        
        # # Create a new filter
        kalman = UncentedKalmanFilter()
        # kalman.initialize()
    
        for k in xrange(1,N):
            z = h(s) + r * np.random.randn()
            sV[:,k] = s.T[0]
            zV[k] = z
            x, P = kalman.step(f, x, P, h, z, Q, R) #step(z, R, f, h)
            # print "x", x
            # print "z", z
            xV[:,k] = x.T[0]
            s = f(s) + q*np.random.randn()
            
        try: # If we have ROOT install try plotting with it
            from ROOT import TCanvas, TGraph, kBlue, kGreen
        
            xVx = TGraph()
            xVy = TGraph()
            xVz = TGraph()
            xVx.SetLineColor(kGreen)
            xVy.SetLineColor(kGreen)
            xVz.SetLineColor(kGreen)
            
            sVx = TGraph()
            sVy = TGraph()
            sVz = TGraph()
            sVx.SetLineColor(kBlue)
            sVy.SetLineColor(kBlue)
            sVz.SetLineColor(kBlue)
        
            zVx = TGraph()
            # zVx.SetLineColor(kBlue)
            zVx.SetLineStyle(3)
            
            for i,j in enumerate(xV[0]):
                zVx.SetPoint(i+1, i+1, zV[i])
                
                xVx.SetPoint(i+1, i+1, xV[0][i])
                xVy.SetPoint(i+1, i+1, xV[1][i])
                xVz.SetPoint(i+1, i+1, xV[2][i])
        
                sVx.SetPoint(i+1, i+1, sV[0][i])
                sVy.SetPoint(i+1, i+1, sV[1][i])
                sVz.SetPoint(i+1, i+1, sV[2][i])
                
            c = TCanvas("unittest")
            c.Divide(1,3)
        
            c.cd(1)
            sVx.Draw("ALP")
            xVx.Draw("LP")
            zVx.Draw("LP")
            
            c.cd(2)
            sVy.Draw("ALP")
            xVy.Draw("LP")
        
            c.cd(3)
            sVz.Draw("ALP")
            xVz.Draw("LP")
                
            raw_input("Done..")
        except:
            print "Failed something with ROOT plotting"
                
        
        
        return self.assertTrue(True)

        
        
        
        
if __name__ == '__main__':
	unittest.main()