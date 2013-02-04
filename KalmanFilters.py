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
    """docstring for UncentedKalmanFilter"""
    def __init__(self, initial_state, initial_covariance, process_noise):
        super(UncentedKalmanFilter, self).__init__()
        self.initial_state = initial_state
        self.initial_covariance = initial_covariance
        self.process_noise = process_noise

        self.has_initialized = False
        self.has_calculated_sigma_points = False
        self.step_i = 0
        
    def calculate_sigma_points(self):
        """Calculate the sigma sampling points"""
        # Estimate sigma points xsig and weights W
        if self.has_initialized and not self.has_calculated_sigma_points:
            self.x_sigma = np.array((1+2*self.Nx ) * [self.x]).T
            self.W = np.array((1+2*self.Nx) *[0.33]) #Random W_0 value see article
            for i in xrange(self.Nx):        
    
                # Calculate the point offsets (eq. 12 in article)
                offset = np.linalg.cholesky((self.Nx/(1-self.W[0]))*self.P)[i].T
                
                self.x_sigma[:,i+1] +=  offset
                self.x_sigma[:,i+self.Nx+1] -= offset
        
                self.W[i+1] = (1.0 - self.W[0]) / (2.0*self.Nx)
                self.W[i+self.Nx+1] = (1.0 - self.W[0]) / (2.0*self.Nx)
                
            self.has_calculated_sigma_points = True
            
    def initialize(self):
        """docstring for initialize"""
        if not self.has_initialized:
            self.P = self.initial_covariance
            self.x = self.initial_state
            self.Nx = len(self.x)
            self.Q = self.process_noise
            
            self.has_initialized = True
            self.calculate_sigma_points()
        
    def unscented_transform(self, func, sigmas, Wm, Wc, n, R):
        """Uncented transformation"""

        L = np.size(sigmas,1)
        y = np.zeros([n,1]).T
        Y = np.zeros([n,L])

        for k in xrange(L):
            Y[:,k] = func(sigmas.T[k])
            y += Wm[k] * Y[:,k] # Calculate weighted average from each sigma point


        Y1 = Y - y.T
        P = np.dot(np.dot(Y1, np.diag(Wc)), Y1.T) + R
        return y,Y,P,Y1
        
    def step(self, measurement, measurement_noise, prediction_function, measurement_function):
        """docstring for compute"""
        if not self.has_initialized:
            self.initialize()
        
        self.z = measurement
        self.Nm = len(self.z)
        
        self.R = measurement_noise
        
        # Run prediction
        x1, X1, P1, X2 = self.unscented_transform(prediction_function, self.x_sigma, self.W, self.W, self.Nx, self.Q)
        
        # Run map to measurement 
        z1, Z1,  P2, Z2 = self.unscented_transform(measurement_function, X1, self.W, self.W, self.Nm, self.R)
        
        P12 = np.dot(np.dot(X2,np.diag(self.W)),Z2.T) # transformed cross-covariance
        K = np.dot(P12, np.linalg.inv(P2)) 
        x = x1 + np.dot( K ,(self.z-z1).T).T # state update
        self.x = x[0] #back to vector form...
        self.P = P1 - np.dot(K,P12.T) # covariance update

        self.r = self.z - z1 # Residual
        print self.r
        print self.z, z1
        print self.x
        
        self.step_i += 1
        
        return self.x, self.P
    
    def propagate(self, measurements, measurement_noise_vector, prediction_function, measurement_function):
        """docstring for propagate"""
        for i,z in enumerate(measurements):
            self.step(z, measurement_noise_vector[i], prediction_function, measurement_function)
            
        return self.x, self.P


class UncentedKalmanFilterTest(unittest.TestCase):
    def setUp(self):
        pass


if __name__ == '__main__':
	unittest.main()