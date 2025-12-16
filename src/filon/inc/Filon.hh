#ifndef Filon_hh
#define Filon_hh

#if defined(_WIN32) || defined(_WIN64)
#ifdef FILON_EXPORTS
#define FILON_EXPORT __declspec(dllexport)
#else
#define FILON_EXPORT __declspec(dllimport)
#endif
#else
#define FILON_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

FILON_EXPORT void cal_integral(unsigned massNum, double temperature,
                               unsigned x_panels, double* xVec, double* yVec,
                               unsigned t_length, double* timeVec,
                               double* gamma_classic,
                               double* gamma_quantum_real,
                               double* gamma_quantum_imag);

FILON_EXPORT void cal_limit(unsigned massNum, double temperature, double* xVec,
                            double* yVec, unsigned t_length, double* timeVec,
                            double* limit_value, double* limit_value_real,
                            double* limit_value_imag);

#ifdef __cplusplus
}
#endif
#endif
