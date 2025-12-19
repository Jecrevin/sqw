#include "Utils.hh"

#include <algorithm>
#include <cmath>
#include <cstring>  //memcpy
#include <iostream>
#include <stdexcept>

#include "constants.hh"

double eKin2k(double eV) { return std::sqrt(eV * const_eV2kk); }

std::pair<double, double> minMaxQ(double enin_eV, double enout_eV) {
  if (enout_eV < 0)
    throw std::runtime_error("enout_eV contains negtive energy");

  double ratio = enout_eV / enin_eV;
  double sqrtR = std::sqrt(ratio);
  double k0 = eKin2k(enin_eV);
  double qmin = k0 * std::sqrt(1. + ratio - 2 * sqrtR);
  double qmax = k0 * std::sqrt(1. + ratio + 2 * sqrtR);
  return std::pair<double, double>(qmin, qmax);
}

double lininterp(double x0, double y0, double x1, double y1, double x) {
  return y0 + (x - x0) * (y1 - y0) / (x1 - x0);
}

unsigned nextGoodFFTNumber(double num) {
  return (unsigned)pow(2, ceil(log2(num)));
}

double stableSum(const std::vector<double>& data) {
  StableSum s;
  for (auto& v : data) {
    s.add(v);
  }
  return s.sum();
}

double stableSum(const double* data, size_t length) {
  StableSum s;
  for (size_t i = 0; i < length; i++) {
    s.add(data[i]);
  }
  return s.sum();
}

void fftshift(double* datainput, size_t length) {
  std::vector<double> data(datainput, datainput + length);
  memcpy(datainput, data.data(), length * sizeof(double));
}

void fftshift(std::vector<double>& data) {
  size_t dataSize = data.size();
  size_t half = dataSize / 2;

  if (dataSize % 2) {
    data.resize(data.size() + 1, data.size());
    std::swap_ranges(data.begin(), data.begin() + half + 1,
                     data.begin() + half + 1);
    data.erase(data.begin() + half);
  } else
    std::swap_ranges(data.begin(), data.begin() + half, data.begin() + half);
}

std::vector<double> fftfreq(unsigned n) {
  std::vector<double> data(n);
  double spacing = 1. / n;
  for (unsigned i = 0; i < n / 2; i++) {
    data[i] = spacing * (i);
  }

  for (unsigned i = 0; i < n / 2; i++) {
    data[i + n / 2] = -0.5 + spacing * (i);
  }
  return data;
}

double trapz(const std::vector<double>& y, const std::vector<double>& x) {
  if (y.size() != x.size()) throw std::runtime_error("trapz size error");

  return trapz(y.data(), x.data(), y.size());
}

double trapz(const double* y, const double* x, size_t length) {
  StableSum s;
  for (unsigned i = 1; i < length; i++) {
    s.add((x[i] - x[i - 1]) * (y[i] + y[i - 1]) * 0.5);
  }
  return s.sum();
}

void flip(const std::vector<double>& arr, std::vector<double>& arr_dst,
          bool opposite_sign) {
  arr_dst.resize(arr.size());
  std::reverse_copy(arr.begin(), arr.end(), arr_dst.begin());

  if (opposite_sign) {
    std::vector<double>::iterator it(arr_dst.begin()), itE(arr_dst.end());
    for (; it != itE; ++it) *it = -(*it);
  }
}

std::vector<double> logspace(double start, double stop, unsigned num) {
  std::vector<double> vec(num);
  double interval = (stop - start) / (num - 1);
  for (std::vector<double>::iterator it = vec.begin(); it != vec.end(); ++it) {
    *it = pow(10, start);
    start += interval;
  }
  return vec;
}

std::vector<double> linspace(double start, double stop, unsigned num) {
  // nc_assert(num>1);
  std::vector<double> vec(num);
  double interval = (stop - start) / (num - 1);
  for (std::vector<double>::iterator it = vec.begin(); it != vec.end(); ++it) {
    *it = start;
    start += interval;
  }
  return vec;
}

std::vector<double> operator*(const std::vector<double>& v, double factor) {
  std::vector<double> result;
  result.reserve(v.size());
  for (auto vv : v) result.push_back(vv * factor);
  return result;
}

void* std_vector_fromArray(unsigned n, void* v) {
  auto doubleVec = static_cast<double*>(v);
  return static_cast<void*>(
      new std::vector<double>(&doubleVec[0], &doubleVec[0] + n));
}

void* std_vector_new() { return static_cast<void*>(new std::vector<double>); }

void std_vector_delete(void* v) { delete static_cast<std::vector<double>*>(v); }

int std_vector_size(void* v) {
  return static_cast<std::vector<double>*>(v)->size();
}

double std_vector_get(void* v, int i) {
  return static_cast<std::vector<double>*>(v)->at(i);
}

void std_vector_push_back(void* v, double i) {
  static_cast<std::vector<double>*>(v)->push_back(i);
}
