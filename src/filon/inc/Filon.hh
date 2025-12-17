#ifndef Filon_hh
#define Filon_hh

#ifdef __cplusplus
extern "C" {
#endif

void sin_integral_single(unsigned x_panels, double* xVec, double* yVec,
                         double time, double& sin_single);

void cos_integral_single(unsigned x_panels, double* xVec, double* yVec,
                         double time, double& cos_single);

void cal_integral(unsigned massNum, double temperature, unsigned x_panels,
                  double* xVec, double* yVec, unsigned t_length,
                  double* timeVec, double* gamma_classic,
                  double* gamma_quantum_real, double* gamma_quantum_imag);

void cal_limit(unsigned massNum, double temperature, double* xVec, double* yVec,
               unsigned t_length, double* timeVec, double* limit_value,
               double* limit_value_real, double* limit_value_imag);

void cal_PDF(double rho, unsigned x_panels, double* xVec, double* yVec,
             unsigned r_length, double* rVec, double* pdfVec);

#ifdef __cplusplus
}
#endif
#endif
