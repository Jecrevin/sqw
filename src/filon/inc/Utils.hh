#ifndef Utils_hh
#define Utils_hh

#include <utility>
#include <vector>  // std::pair, std::make_pair

double eKin2k(double eV);
std::pair<double, double> minMaxQ(double enin_eV, double enout_eV);

double lininterp(double x0, double y0, double x1, double y1, double x);
unsigned nextGoodFFTNumber(double num);
double stableSum(const std::vector<double>& data);
double stableSum(const double* data, size_t length);
// numpy like functions
void fftshift(std::vector<double>& data);
void fftshift(double* data, size_t length);

std::vector<double> fftfreq(unsigned n);
double trapz(const std::vector<double>& y, const std::vector<double>& x);
double trapz(const double* y, const double* x, size_t length);

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
// wrapper of std::vector in C and python
void* std_vector_fromArray(unsigned n, void* v);
void* std_vector_new();
void std_vector_delete(void* v);
int std_vector_size(void* v);
double std_vector_get(void* v, int i);
void std_vector_push_back(void* v, double i);

#ifdef __cplusplus
}
#endif

#include "Utils.icc"
#endif
