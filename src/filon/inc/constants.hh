#ifndef constexprants_hh
#define constexprants_hh

#include <cmath>

constexpr double kAngstrom = 1.;
constexpr double kSecond = 1.;
constexpr double kPicoSecond = kSecond * 1e-12;
constexpr double kfemtoSecond = kSecond * 1e-15;
constexpr double kPlanck = 4.13566769692386e-15;  //(source: NIST/CODATA 2018);
constexpr double kHbar = kPlanck * 0.5 / M_PI;    //[eV*s];
constexpr double kEV2radpsec = 1. / kHbar;
constexpr double kRadpsec2meV = kHbar * 1e3;
constexpr double kRadpfs2meV = kRadpsec2meV * 1e15;

constexpr double kDeg2rad = M_PI / 180.;
constexpr double kEV2kk = 1 / 2.072124652399821e-3;
constexpr double kC = 299792458e10;  // speed of light in Aa/s;
constexpr double kDalton2kg =
    1.660539040e-27;  // amu to kg (source: NIST/CODATA 2018);
constexpr double kDalton2eVc2 =
    931494095.17;  // amu to eV/c^2 (source: NIST/CODATA 2018);
constexpr double kAvogadro =
    6.022140857e23;  // mol^-1 (source: NIST/CODATA 2018);
constexpr double kBoltzmann = 8.6173303e-5;                   // eV/K;
constexpr double kDalton2TakMass = kDalton2eVc2 / (kC * kC);  // eV/(As/s)^2
constexpr double kNeutronAtomicMass = 1.00866491588;          // atomic unit;
constexpr double kNeutronMassEvc2 =
    kNeutronAtomicMass * kDalton2TakMass;  // eV/(Aa/s)^2  ;
constexpr double kEkin2v = sqrt(
    2.0 /
    kNeutronMassEvc2);  // multiply with sqrt(ekin) to get velocity in Aa/s;

constexpr double const_eV2kk = 1 / 2.072124652399821e-3;
#endif
