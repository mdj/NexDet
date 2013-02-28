#!/usr/bin/env python
# encoding: utf-8
"""
NumericSolvers.py

Created by Morten Dam Jørgensen on 2013-02-07.
Copyright (c) 2013 Niels Bohr Institute, Copenhagen. All rights reserved.
"""

import sys
import os
import unittest


class NumericSolvers:
    def __init__(self):
        pass



    def euler(self):
        """Euler integration of x = dx * f(t,x)"""
        pass
    def runge_kutta(self):
        """4'order runge kutta"""
        pass
        
    def adaptive_runge_kutta(self):
        """docstring for adaptive_runge_kutta"""
        pass


#################### OLD CODE ##############################################
    def Fmag(x, p, q, m):
        """docstring for Bfield"""
        return (kappa*q) * np.cross(p/m, getBfield(x))

    def Vmag(x,p, q, m):
        """docstring for Vmag"""
        return p/m

    def rk4(x, p, dt):
        """Runge-Kutta 4'th order integration"""
        k1p = dt * Fmag(x,p, q, m)
        k1x = dt * Vmag(x,p, q, m)

        k2p = dt * Fmag(x + dt/2.0,p + 0.5*k1p, q, m)
        k2x = dt * Vmag(x + 0.5*k1x, p + dt/2.0, q, m)

        k3p = dt * Fmag(x + dt/2.0,p + 0.5*k2p, q, m)
        k3x = dt * Vmag(x + 0.5*k2x, p + dt/2.0, q, m)

        k4p = dt * Fmag(x + dt,p + k3p, q, m)
        k4x = dt * Vmag(x + k3x, p, q, m)

        p = p + 1.0/6.0 * (k1p + 2*k2p + 2*k3p + k4p) # RK4        
        x = x + 1.0/6.0 * (k1x + 2*k2x + 2*k3x + k4x) # RK4

        return x,p

    def euler_step(r, dz, z):
        """Euler integration but in proper 5-elment format"""
        x = r[0]
        y = r[1]
        tx = r[2]
        ty = r[3]
        qoverp = r[4]

        B = getBfield([x,y,z])
        Bx = B[0]
        By = B[1]
        Bz = B[2]

        dx = dz * tx
        dy = dz * ty

        ds = dz * kappa * qoverp * sqrt(1+tx**2 + ty**2)
        dtx = ds * (tx*ty*Bx - (1+tx**2)*By + ty*Bz)
        dty = ds * ((1+ty**2)*Bx -tx*ty*By - tx*Bz)

        dqoverp = dz * 0.0

        return r + np.array([dx,dy,dtx,dty,dqoverp])


    def rk_45_step(z, r, dz=0.001):
        """
        Calculate the Runge-Kutta 4'th degree step of the 5-element formatted version
        """

        def f(z, r):
            """Euler integration but in proper 5-elment format"""

            # print 10*"-"            
            # print "In function term"
            # print z, r
            x = r[0]
            y = r[1]
            tx = r[2]
            ty = r[3]
            qoverp = r[4]

            B = getBfield([x,y,z])
            # print "Bfield", B
            Bx = B[0]
            By = B[1]
            Bz = B[2]

            dx = tx
            dy = ty

            dtx = kappa * qoverp * sqrt(1+tx**2 + ty**2) * (tx*ty*Bx - (1+tx**2)*By + ty*Bz)
            dty = kappa * qoverp * sqrt(1+tx**2 + ty**2) * ((1+ty**2)*Bx -tx*ty*By - tx*Bz)
            dqoverp =  0.0

            # print np.array([dx,dy,dtx,dty,dqoverp])
            # print 10*"-"
            return np.array([dx,dy,dtx,dty,dqoverp])



        # Runge-Kutta-Fehlberg method
        # Returns t,x, and the single step trunctation error, eps
        # coefficients
        c20=0.25
        c21=0.25
        c30=0.375
        c31=0.09375
        c32=0.28125
        c40=12/13.0
        c41=1932/2197.0
        c42= -7200/2197.0
        c43=7296/2197.0
        c51=439/216.0
        c52= -8.0
        c53=3680/513.0
        c54=-845/4104.0
        c60=0.5
        c61= -8/27.0
        c62=2.0
        c63= -3544/2565.0
        c64=1859/4104.0
        c65= -0.275
        a1=25/216.0
        a2=0.0
        a3=1408/2565.0
        a4=2197/4104.0
        a5= -0.2
        b1=16/135.0
        b2=0.0
        b3=6656/12825.0
        b4=28561/56430.0
        b5= -0.18
        b6=2/55.0
        # K values
        K1 = dz*f(z,r);
        K2 = dz*f(z+c20*dz, r+c21*K1);
        K3 = dz*f(z+c30*dz, r+c31*K1+c32*K2);
        K4 = dz*f(z+c40*dz, r+c41*K1+c42*K2+c43*K3);
        K5 = dz*f(z+dz,r+c51*K1+c52*K2+c53*K3+c54*K4);
        K6 = dz*f(z+c60*dz,r+c61*K1+c62*K2+c63*K3+c64*K4+c65*K5);
        r4 = r + a1*K1 + a3*K3 + a4*K4 + a5*K5; #RK4
        r = r + b1*K1 + b3*K3 + b4*K4 + b5*K5 + b6*K6; #RK5
        z = z+dz;
        eps = abs(r - r4)

        return z, r, np.amax(eps)

    def adpative_runge_kutta(z,zf,r):
        """Integrate from z to zf"""

        dz = 0.0001
        emax = 1.0e-6
        emin = 1.0e-16
        dzmin = 0.00001
        dzmax = 0.1
        iflag = 1
        delta = 0.5e-5

        r_sol = []
        z_sol = []
        nitr = 0
        while 1: # Set iteration max here potentially
            nitr += 1
            r0 = r
            z0 = z

            if abs(dz) < dzmin:
                dz = np.sign(dz)*dzmin
            if abs(dz) > dzmax:
                dz = np.sign(dz)*dzmax

            d = abs(zf - z)
            if d <= abs(dz):
                iflag = 0 # terminate at next round
                if d <= delta*max(abs(zf), abs(z)): break

            # Call integrator
            z, r, eps = rk_45_step(z, r, dz)

            if eps < emin:
                dz = 2.0*dz
            if eps > emax:
                dz = dz/2.0
                z = z0 # Redo the previous step with the finer resolution
                r = r0
                continue

            r_sol.append(r)
            z_sol.append(z)
            if iflag == 0: break

        return z_sol, r_sol, nitr



class NumericSolversTests(unittest.TestCase):
    def setUp(self):
        pass


if __name__ == '__main__':
    unittest.main()