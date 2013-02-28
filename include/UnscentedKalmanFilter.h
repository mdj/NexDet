#ifndef UNSCENTEDKALMANFILTER_DAMCONSULT
#define UNSCENTEDKALMANFILTER_DAMCONSULT


// class TMatrixD;

class UnscentedKalmanFilter {  

public:
    
    TMatrixD sigmas(TMatrixD M, TMatrixD P, double c);

           
};

#endif /* end of include guard: UNSCENTEDKALMANFILTER_DAMCONSULT */
