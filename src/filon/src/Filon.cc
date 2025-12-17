#include "Filon.hh"

#include <math.h>
#include <stdlib.h>

#include <iostream>

#include "ProgressMonitor.hh"
#include "Utils.hh"
#include "omp.h"

constexpr double i_105 = 1. / 105;
constexpr double i_15 = 1. / 15;
constexpr double beta_limit = 2. / 3;
constexpr double i_210 = 1. / 210;
constexpr double gamma_limit = 4. / 3;
constexpr double i_4725 = 1. / 4725;
constexpr double i_315 = 1. / 315;
constexpr double i_45 = 1. / 45;
constexpr long double const_planck = 4.13566769692386e-15L;
constexpr long double const_hbar = const_planck * 0.5 / M_PI;
constexpr long double const_neutron_mass_evc2 = 1.0454075098625835e-28L;
constexpr double const_boltzmann = 8.6173303e-5L;

static void s2p_1_s2p(unsigned x_panels, double* xVec, double* yVec,
                      unsigned t_length, double* tVec, double* s2p_1_Vec,
                      double* s2p_Vec) {
#pragma omp parallel for simd
  for (auto i = 0; i < t_length; i++) {
    double time = tVec[i];
    StableSum sum1, sum2;
    // double sum1=0.0;
    // double sum2=0.0;
    double part_2 =
        -0.5 * (yVec[0] * sin(time * xVec[0]) +
                yVec[2 * x_panels] * sin(time * xVec[2 * x_panels]));
    for (auto j = 1; j < x_panels + 1; j++) {
      sum1.add(yVec[2 * j - 1] * sin(time * xVec[2 * j - 1]));
      sum2.add(yVec[2 * (j - 1)] * sin(time * xVec[2 * (j - 1)]));
    }
    sum2.add(yVec[2 * x_panels] * sin(time * xVec[2 * x_panels]));
    s2p_1_Vec[i] = sum1.sum();
    s2p_Vec[i] = sum2.sum() + part_2;
  }
}

static void c2p_1_c2p(unsigned x_panels, double* xVec, double* yVec,
                      unsigned t_length, double* tVec, double* c2p_1_Vec,
                      double* c2p_Vec) {
#pragma omp parallel for simd
  for (auto i = 0; i < t_length; i++) {
    double time = tVec[i];
    StableSum sum1, sum2;
    // double sum1=0.0;
    // double sum2=0.0;
    double part_2 =
        -0.5 * (yVec[0] * cos(time * xVec[0]) +
                yVec[2 * x_panels] * cos(time * xVec[2 * x_panels]));
    for (auto j = 1; j < x_panels + 1; j++) {
      sum1.add(yVec[2 * j - 1] * cos(time * xVec[2 * j - 1]));
      sum2.add(yVec[2 * (j - 1)] * cos(time * xVec[2 * (j - 1)]));
    }
    sum2.add(yVec[2 * x_panels] * cos(time * xVec[2 * x_panels]));
    c2p_1_Vec[i] = sum1.sum();
    c2p_Vec[i] = sum2.sum() + part_2;
    // c2p_Vec[i]=sum2;
  }
}

void alpha_beta_gamma(unsigned theta_length, double* thetaVec, double* alphaVec,
                      double* betaVec, double* gammaVec) {
  double theta = 0.0;
#pragma omp parallel for simd
  for (auto i = 0; i < theta_length; i++) {
    theta = thetaVec[i];
    double theta2 = theta * theta;
    double theta3 = theta2 * theta;
    double theta4 = theta3 * theta;

    double itheta3 = 1. / theta3;
    if (theta < 0.0218) {
      betaVec[i] = -4. * theta4 * i_105 + 2. * theta2 * i_15 + beta_limit;
    } else {
      double cos2_theta = cos(theta) * cos(theta);
      double sin_2theta = sin(2 * theta);
      betaVec[i] = 2 * itheta3 * (theta * (1 + cos2_theta) - sin_2theta);
    }
    if (theta < 0.0365) {
      gammaVec[i] = theta4 * i_210 - 2. * theta2 * i_15 + gamma_limit;
    } else {
      gammaVec[i] = 4. * itheta3 * (sin(theta) - theta * cos(theta));
    }
    if (theta < 0.086) {
      double theta7 = theta3 * theta4;
      double theta5 = theta4 * theta;
      alphaVec[i] =
          2. * theta7 * i_4725 - 2. * theta5 * i_315 + 2. * theta3 * i_45;
    } else {
      double sin_2theta = sin(2 * theta);
      double sin2_theta = sin(theta) * sin(theta);
      alphaVec[i] =
          itheta3 * (theta2 + theta * sin_2theta * 0.5 - 2 * sin2_theta);
    }
  }
}

void sin_integral(unsigned x_panels, double* xVec, double* yVec,
                  unsigned t_length, double* tVec, double* sinVec) {
  double h = xVec[1] - xVec[0];
  double* theta = new double[t_length];
  for (auto i = 0; i < t_length; i++) {
    theta[i] = tVec[i] * h;
  }
  double* alpha = new double[t_length];
  double* beta = new double[t_length];
  double* gamma = new double[t_length];
  double* s2p_1 = new double[t_length];
  double* s2p = new double[t_length];

  alpha_beta_gamma(t_length, theta, alpha, beta, gamma);
  s2p_1_s2p(x_panels, xVec, yVec, t_length, tVec, s2p_1, s2p);

  int lastIdx = 2 * x_panels;
#pragma omp parallel for simd
  for (auto i = 0; i < t_length; i++) {
    double part_1 = alpha[i] * (yVec[0] * cos(tVec[i] * xVec[0]) -
                                yVec[lastIdx] * cos(tVec[i] * xVec[lastIdx]));
    double part_2 = beta[i] * s2p[i];
    double part_3 = (gamma[i]) * s2p_1[i];
    sinVec[i] = h * (part_1 + part_2 + part_3);
  }
  delete[] alpha;
  delete[] beta;
  delete[] gamma;
  delete[] s2p_1;
  delete[] s2p;
}

void cos_integral(unsigned x_panels, double* xVec, double* yVec,
                  unsigned t_length, double* tVec, double* cosVec) {
  double h = xVec[1] - xVec[0];
  double* theta = new double[t_length];
  for (auto i = 0; i < t_length; i++) {
    theta[i] = tVec[i] * h;
  }
  double* alpha = new double[t_length];
  double* beta = new double[t_length];
  double* gamma = new double[t_length];
  double* c2p_1 = new double[t_length];
  double* c2p = new double[t_length];

  alpha_beta_gamma(t_length, theta, alpha, beta, gamma);
  c2p_1_c2p(x_panels, xVec, yVec, t_length, tVec, c2p_1, c2p);

  int lastIdx = 2 * x_panels;
#pragma omp parallel for simd
  for (auto i = 0; i < t_length; i++) {
    // std::cout<<alpha[i]<<";"<<beta[i]<<";"<<gamma[i]<<std::endl;
    double part_1 = alpha[i] * (yVec[lastIdx] * sin(tVec[i] * xVec[lastIdx]) -
                                yVec[0] * sin(tVec[i] * xVec[0]));
    double part_2 = beta[i] * c2p[i];
    double part_3 = abs(gamma[i]) * c2p_1[i];
    cosVec[i] = h * (part_1 + part_2 + part_3);
  }
  delete[] alpha;
  delete[] beta;
  delete[] gamma;
  delete[] c2p_1;
  delete[] c2p;
}

void alpha_beta_gamma_single(double theta, double& alpha, double& beta,
                             double& gamma) {
  double theta2 = theta * theta;
  double theta3 = theta2 * theta;
  double theta4 = theta3 * theta;

  double itheta3 = 1. / theta3;
  if (theta < 0.0218) {
    beta = -4. * theta4 * i_105 + 2. * theta2 * i_15 + beta_limit;
  } else {
    double cos2_theta = cos(theta) * cos(theta);
    double sin_2theta = sin(2 * theta);
    beta = 2 * itheta3 * (theta * (1 + cos2_theta) - sin_2theta);
  }
  if (theta < 0.0365) {
    gamma = theta4 * i_210 - 2. * theta2 * i_15 + gamma_limit;
  } else {
    gamma = 4. * itheta3 * (sin(theta) - theta * cos(theta));
  }
  if (theta < 0.086) {
    double theta7 = theta3 * theta4;
    double theta5 = theta4 * theta;
    alpha = 2. * theta7 * i_4725 - 2. * theta5 * i_315 + 2. * theta3 * i_45;
  } else {
    double sin_2theta = sin(2 * theta);
    double sin2_theta = sin(theta) * sin(theta);
    alpha = itheta3 * (theta2 + theta * sin_2theta * 0.5 - 2 * sin2_theta);
  }
}
void s2p_1_s2p_single(unsigned x_panels, double* xVec, double* yVec,
                      double time, double& s2p_1, double& s2p) {
  // std::cout<<"start s2p_single"<<std::endl;
  double sum1 = 0.0;
  double sum2 = 0.0;
  double part_2 = -0.5 * (yVec[0] * sin(time * xVec[0]) +
                          yVec[2 * x_panels] * sin(time * xVec[2 * x_panels]));
  // std::cout<<"part_2"<<part_2<<std::endl;
  // #pragma omp parallel for simd
  // this parallel effect the results
  for (auto j = 1; j < x_panels + 1; j++) {
    sum1 += yVec[2 * j - 1] * sin(time * xVec[2 * j - 1]);
    sum2 += yVec[2 * (j - 1)] * sin(time * xVec[2 * (j - 1)]);
  }

  sum2 += yVec[2 * x_panels] * sin(time * xVec[2 * x_panels]);
  // std::cout<<"sum2 final:"<<sum2<<std::endl;
  s2p_1 = sum1;
  s2p = sum2 + part_2;
  // std::cout<<s2p_1<<";"<<s2p<<std::endl;
}
void c2p_1_c2p_single(unsigned x_panels, double* xVec, double* yVec,
                      double time, double& c2p_1, double& c2p) {
  double sum1 = 0.0;
  double sum2 = 0.0;
  double part_2 = -0.5 * (yVec[0] * cos(time * xVec[0]) +
                          yVec[2 * x_panels] * cos(time * xVec[2 * x_panels]));
  // #pragma omp parallel for simd
  for (auto j = 1; j < x_panels + 1; j++) {
    sum1 += yVec[2 * j - 1] * cos(time * xVec[2 * j - 1]);
    sum2 += yVec[2 * (j - 1)] * cos(time * xVec[2 * (j - 1)]);
  }
  sum2 += yVec[2 * x_panels] * cos(time * xVec[2 * x_panels]);
  c2p_1 = sum1;
  c2p = sum2 + part_2;
}

void sin_integral_single(unsigned x_panels, double* xVec, double* yVec,
                         double time, double& sin_single) {
  double h = xVec[1] - xVec[0];
  double theta = h * time;
  double alpha = 0.0;
  double beta = 0.0;
  double gamma = 0.0;
  alpha_beta_gamma_single(theta, alpha, beta, gamma);
  // std::cout<<"beta"<<beta<<std::endl;
  double s2p_1 = 0.0;
  double s2p = 0.0;
  s2p_1_s2p_single(x_panels, xVec, yVec, time, s2p_1, s2p);
  // std::cout<<"s2p"<<s2p<<std::endl;
  int lastIdx = 2 * x_panels;
  double part_1 = alpha * (yVec[0] * cos(time * xVec[0]) -
                           yVec[lastIdx] * cos(time * xVec[lastIdx]));
  double part_2 = beta * s2p;
  double part_3 = gamma * s2p_1;
  // std::cout<<"p1p2p3: "<<part_1<<part_2<<part_3<<std::endl;
  sin_single = h * (part_1 + part_2 + part_3);
}

void cos_integral_single(unsigned x_panels, double* xVec, double* yVec,
                         double time, double& cos_single) {
  double h = xVec[1] - xVec[0];
  double theta = h * time;
  double alpha = 0.0;
  double beta = 0.0;
  double gamma = 0.0;
  alpha_beta_gamma_single(theta, alpha, beta, gamma);
  double c2p_1 = 0.0;
  double c2p = 0.0;
  c2p_1_c2p_single(x_panels, xVec, yVec, time, c2p_1, c2p);
  int lastIdx = 2 * x_panels;
  double part_1 = alpha * (yVec[lastIdx] * sin(time * xVec[lastIdx]) -
                           yVec[0] * sin(time * xVec[0]));
  double part_2 = beta * c2p;
  double part_3 = gamma * c2p_1;
  cos_single = h * (part_1 + part_2 + part_3);
}

// g(r)-1 part
void gr_func(double rho, unsigned x_panels, double* xVec, double* yVec,
             double r, double* fVec) {
  // #pragma omp parallel for simd
  for (auto j = 0; j < (2 * x_panels + 1); j++) {
    double qr = xVec[j] * r;
    fVec[j] = 0.5 / (M_PI * M_PI) / rho / r * xVec[j] * yVec[j];
  }
}

// cal g(r)-1
void cal_PDF(double rho, unsigned x_panels, double* xVec, double* yVec,
             unsigned r_length, double* rVec, double* pdfVec) {
  // ProgressMonitor* moni = new ProgressMonitor("integralGamma", r_length,
  // 0.1);
  double* fVec = new double[2 * x_panels + 1];
  for (auto i = 0; i < r_length; i++) {
    gr_func(rho, x_panels, xVec, yVec, rVec[i], fVec);
    // std::cout<<"fvec: "<<fVec[0]<<std::endl;
    sin_integral_single(x_panels, xVec, fVec, rVec[i], pdfVec[i]);
    // std::cout<<"pdfvec: "<<pdfVec[0]<<std::endl;
    // moni->OneTaskCompleted();
  }
  delete[] fVec;
}

// gamma part
void gamma_func(unsigned massNum, double temperature, unsigned x_panels,
                double* xVec, double* yVec, double time, double* fVec_cls,
                double* fVec_real, double* fVec_imag) {
  /***
  define gamma function
  quantum:
  hbar/mass*dos/omega*{coth(0.5*hbar*omage/kt)*(1-cos(omega*t)-sin(omega*t)j)}
  classic: 2/mass*kt*dos/omega^2*(1-cos(omage*t))
  ***/
  double i_mbeta =
      const_boltzmann * temperature / (const_neutron_mass_evc2 * massNum);
  double hbar_m = const_hbar / (const_neutron_mass_evc2 * massNum);
  double hbarbeta = const_hbar / const_boltzmann / temperature;
#pragma omp parallel for simd
  for (auto j = 0; j < (2 * x_panels + 1); j++) {
    double i_omega_2 = 1. / (xVec[j] * xVec[j]);
    long double omegat = xVec[j] * time;
    fVec_cls[j] = 4 * i_mbeta * yVec[j] * i_omega_2 * sin(omegat * 0.5);
    fVec_real[j] = hbar_m * yVec[j] / xVec[j] * 1. /
                   tanh(hbarbeta * xVec[j] * 0.5) * 2 * sin(omegat * 0.5);
    fVec_imag[j] = hbar_m * yVec[j] / xVec[j];
  }
}

void cal_integral(unsigned massNum, double temperature, unsigned x_panels,
                  double* xVec, double* yVec, unsigned t_length,
                  double* timeVec, double* gamma_classic,
                  double* gamma_quantum_real, double* gamma_quantum_imag) {
  /***
  integral range: omege[1:]
  **/
  ProgressMonitor* moni = new ProgressMonitor("integralGamma", t_length, 0.01);
  double* fVec_cls = new double[2 * x_panels + 1];
  double* fVec_real = new double[2 * x_panels + 1];
  double* fVec_imag = new double[2 * x_panels + 1];
  for (auto i = 0; i < t_length; i++) {
    gamma_func(massNum, temperature, x_panels, xVec, yVec, timeVec[i], fVec_cls,
               fVec_real, fVec_imag);
    sin_integral_single(x_panels, xVec, fVec_cls, timeVec[i] * 0.5,
                        gamma_classic[i]);
    sin_integral_single(x_panels, xVec, fVec_real, timeVec[i] * 0.5,
                        gamma_quantum_real[i]);
    sin_integral_single(x_panels, xVec, fVec_imag, timeVec[i],
                        gamma_quantum_imag[i]);
    moni->OneTaskCompleted();
  }
  delete[] fVec_cls;
  delete[] fVec_real;
  delete[] fVec_imag;
}

void cal_limit(unsigned massNum, double temperature, double* xVec, double* yVec,
               unsigned t_length, double* timeVec, double* limit_value,
               double* limit_value_real, double* limit_value_imag) {
  /***
  limit range: interal from omege[0] to omega[1]
  **/
  double i_mbeta =
      const_boltzmann * temperature / (const_neutron_mass_evc2 * massNum);
  double hbar_m = const_hbar / (const_neutron_mass_evc2 * massNum);
  double hbarbeta = const_hbar / const_boltzmann / temperature;
  double step = xVec[1] - xVec[0];

  // size_t numPonit = 100;
  // std::vector<double> func(numPonit);
  // std::vector<double> omega(numPonit);
  // #pragma omp parallel for simd
  for (auto i = 0; i < t_length; i++) {
    double f0_cls = yVec[0] * i_mbeta * timeVec[i] * timeVec[i];
    double f0_real = f0_cls;
    double f0_imag = hbar_m * yVec[0] * timeVec[i];

    long double t2 = timeVec[i] * timeVec[i];
    long double t3 = t2 * timeVec[i];
    long double t5 = t2 * t3;

    double sinVal = sin(xVec[1] * timeVec[i] * 0.5);
    double f1_cls =
        4 * yVec[1] * i_mbeta / (xVec[1] * xVec[1]) * sinVal * sinVal;
    double f1_real = hbar_m * 2 * yVec[1] / xVec[1] * sinVal * sinVal * 1. /
                     tanh(0.5 * hbarbeta * xVec[1]);
    double f1_imag = hbar_m * yVec[1] / xVec[1] * sin(xVec[1] * timeVec[i]);
    // std::cout<<timeVec[i]<<"||"<<-f0_imag<<"||"<<-f1_imag<<"||"<<f1_imag/f0_imag<<std::endl;
    limit_value[i] = (f0_cls + f1_cls) * step * 0.5;
    limit_value_real[i] = (f0_real + f1_real) * step * 0.5;
    limit_value_imag[i] = (f0_imag + f1_imag) * step * 0.5;
    // std::cout<<timeVec[i]<<"||"<<contribution<<"||"<<limit_value_imag[i]<<std::endl;
    // limit_value_imag[i]=contribution;
  }
}
