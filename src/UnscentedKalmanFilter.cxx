#ifndef __CINT__
#include "Riostream.h"
#include "TMath.h"
#include "TMatrixD.h"
#include "TMatrixDLazy.h"
#include "TVectorD.h"
#include "TDecompLU.h"
#include "TDecompSVD.h"
#include "TDecompChol.h"
#endif

#include "../include/UnscentedKalmanFilter.h"

TMatrixD
UnscentedKalmanFilter::sigmas(TMatrixD M, TMatrixD P, double c)
{
    /*
    %UT_SIGMAS - Generate Sigma Points for Unscented Transformation
    %
    % Syntax:
    %   X = ut_sigmas(M,P,c);
    %
    % In:
    %   M - Initial state mean (Nx1 column vector)
    %   P - Initial state covariance
    %   c - Parameter returned by UT_WEIGHTS
    %
    % Out:
    %   X - Matrix where 2N+1 sigma points are as columns
    %
    % Description:
    %   Generates sigma points and associated weights for Gaussian
    %   initial distribution N(M,P). For default values of parameters
    %   alpha, beta and kappa see UT_WEIGHTS.
    %
    % See also UT_WEIGHTS UT_TRANSFORM UT_SIGMAS
    %

    % Copyright (C) 2006 Simo Särkkä
    %
    % $Id: ut_sigmas.m 109 2007-09-04 08:32:58Z jmjharti $
    %
    % This software is distributed under the GNU General Public
    % Licence (version 2 or later); please refer to the file
    % Licence.txt, included with the software, for details.

    function X = ut_sigmas(M,P,c);

    %  A = schol(P);
      A = chol(P)';
      X = [zeros(size(M)) A -A];
      X = sqrt(c)*X + repmat(M,1,size(X,2));
    */
      TDecompChol Cho(P);
      if (!Cho.Decompose())
          printf("Failed to decompose P\n");
      else {
          TMatrixD A = Cho.GetU();
          TMatrixD X(M.GetNrows(),1+2*A.GetNcols());
          X.Zero();
          X.SetSub(0,1,A);
          // X.SetSub(0,A.GetNcols()+1,-1*A);
          
          X.Print();
          TMatrixD Xx(M.GetNrows(),1+2*A.GetNcols());
          
          for(size_t i = 0; i < Xx.GetNcols(); ++i)
          {
              Xx.SetSub(0,i,M);
          }
          
          X *= sqrt(c);
          X += Xx;
          X.Print();
          return X;
          
      }
           
      return TMatrixD(1,1);
    
}
           