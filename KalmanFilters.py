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
            
    def initialize(self, reinitialize = False):
        """docstring for initialize"""
        if not self.has_initialized or reinitialize:
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
        

        # Create a new filter
        kalman = UncentedKalmanFilter(self.x, self.P, self.Q)
        kalman.initialize()
        self.assertTrue(True)

    def test_stepping(self):
        """Run the kalman filter for one step"""
        
        kalman = UncentedKalmanFilter(self.x, self.P, self.Q)
        
        c = 299792458.0 # m/s speed of light
        kappa = 1e-8 * c  # GeV/c T-1 m-1
        
        def getBfield(pos):
            """Return B-field in Tesla for a given position"""
            if pos[2] > 0.3:
                return np.array([1.0, 0.0, 0.0]).T
            else:
                return np.array([0.0, 0.0, 0.0]).T

        def Fmag(x, p, q, m):
            """docstring for Bfield"""
            return (kappa*q) * np.cross(p/m, getBfield(x))

        def Vmag(x,p, q, m):
            """docstring for Vmag"""
            return p/m


        def reco_rk4(param, dt=0.0001):
            """Runge-Kutta 4'th order integration"""


            # FIXME: tell when we have reached another measurement... ds?

            x = param[0:3]
            p = param[3:6]
            q = param[6]
            m = 0.001 # mass assumption
            t = 0
            ts = 0.02

            while t < ts:
                k1p = dt * Fmag(x,p, q, m)
                k1x = dt * Vmag(x,p, q, m)

                k2p = dt * Fmag(x + dt/2.0,p + 0.5*k1p, q, m)
                k2x = dt * Vmag(x + 0.5*k1x, p + dt/2.0, q, m)

                k3p = dt * Fmag(x + dt/2.0,p + 0.5*k2p, q, m)
                k3x = dt * Vmag(x + 0.5*k2x, p + dt/2.0, q, m)

                k4p = dt * Fmag(x + dt,p + k3p, q, m)
                k4x = dt * Vmag(x + k3x, p, q, m)

                dp = 1.0/6.0 * (k1p + 2*k2p + 2*k3p + k4p) # RK4
                dx = 1.0/6.0 * (k1x + 2*k2x + 2*k3x + k4x) # RK4
                p = p + dp
                x = x + dx
                t += dt

            return np.array([x[0], x[1], x[2], p[0], p[1], p[2], q]).T


        def predict_to_measurement(param):
            """Runge-Kutta 4'th order integration"""
            return np.array([param[0], param[1], param[2]])

        
        
        # Define measurements
        measurements = [
                    np.array([-0.00041008907936890824, -0.00090066040845029503, 0.25787750000000004]),
                    np.array([-0.00048417645399443179, -0.0001689257930013671, 0.33607749999999997]), 
                    np.array([-0.00075518418439418286, 0.00607521176150302, 0.62207750000000006])
                    ]                

        measurement_noise = len(measurements) * [0.1 * np.eye(len(measurements[0]))] # Create a noise matrix for each measurement

        # Run the propagator
        x_final, P_final = kalman.propagate(measurements, measurement_noise, reco_rk4, predict_to_measurement)
        
        # todo compare x_final to the correct value and return assertion
        
if __name__ == '__main__':
	unittest.main()