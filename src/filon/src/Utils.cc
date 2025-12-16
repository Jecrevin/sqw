#include "Utils.hh"

#include <algorithm>
#include <cmath>
#include <cstring>  //memcpy
#include <iostream>
#include <stdexcept>

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
