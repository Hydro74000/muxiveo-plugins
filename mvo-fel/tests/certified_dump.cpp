// SPDX-License-Identifier: LGPL-2.1-or-later
// Valeurs de l'EOTF certifiée (chemin double et double-double) pour tests/certified_eotf.py :
// une ligne par signal x : x, L rapide, κ rapide, L double-double (hi, lo), κ double-double.
#include "certified.hpp"
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace muxiveo::fel;
int main() {
    std::vector<double> x;
    // Voisinage du zéro de l'EOTF, échelle logarithmique jusqu'au pôle, et tirages uniformes.
    for (int k = 1; k <= 14; ++k) x.push_back(certified::PQ_ZERO_HIGH*(1 + std::pow(10.0, -k)));
    for (int k = 0; k <= 2000; ++k) x.push_back(certified::PQ_ZERO_HIGH*std::pow(1.95/certified::PQ_ZERO_HIGH, k/2000.0));
    std::mt19937_64 random(5);
    std::uniform_real_distribution<double> unit(0, 1);
    for (int k = 0; k < 2000; ++k) x.push_back(unit(random));
    for (int k = 0; k < 500; ++k) x.push_back(1.9 + 0.09*unit(random));
    std::vector<double> fast = x, kappa(x.size());
    pq_eotf_kappa(fast.data(), kappa.data(), int(x.size()));
    for (size_t i = 0; i < x.size(); ++i) {
        double k;
        const auto value = certified::eotf({x[i], 0}, k);
        std::printf("%a %a %a %a %a %a\n", x[i], fast[i], kappa[i], value.hi, value.lo, k);
    }
}
