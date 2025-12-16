#ifndef Utils_hh
#define Utils_hh

#include <cstddef>
#include <utility>
#include <vector>  // std::pair, std::make_pair

// Stable summation helpers used by Filon integration
double stableSum(const std::vector<double>& data);
double stableSum(const double* data, size_t length);

void flip(const std::vector<double>& arr, std::vector<double>& arr_dst,
          bool opposite_sign);
std::vector<double> logspace(double start, double stop, unsigned num);
std::vector<double> linspace(double start, double stop, unsigned num);

std::vector<double> operator*(const std::vector<double>& v, double factor);

class StableSum {
 public:
  // Numerically stable summation, based on Neumaier's
  // algorithm (doi:10.1002/zamm.19740540106).
  StableSum();
  ~StableSum();
  void add(double x);
  void by(double x);
  double sum() const;
  void clear();

 private:
  double m_sum, m_correction;
};

#ifdef __cplusplus
extern "C" {
#endif
void* std_vector_fromArray(unsigned n, void* v);
void* std_vector_new();
#ifdef __cplusplus
}
#endif

#include "Utils.icc"
#endif
