// SPDX-License-Identifier: LGPL-2.1-or-later
// Reprises certifiées en double-double (voir certified.hpp). Unité compilée avec
// -ffp-contract=off : les transformations exactes (two_sum, produit de Dekker) exigent que
// chaque opération soit arrondie séparément.
#include "certified.hpp"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

static_assert(FLT_EVAL_METHOD == 0, "Arithmétique double sans précision étendue requise");

namespace muxiveo::fel::certified {
namespace {
constexpr double c1 = 3424.0/4096, c2 = 2413.0/128, c3 = 2392.0/128;
constexpr double m1m2 = 6585030.0/524288;  // m1·m2 = (2610/16384)·(2523/32), exact
constexpr double u = 0x1p-100;              // unité d'arrondi majorée des opérations double-double
constexpr DD ln2{0x1.62e42fefa39efp-1, 0x1.abc9e3b39803fp-56};

DD two_sum(double a, double b) {
    const double s = a + b, v = s - a;
    return {s, (a - (s - v)) + (b - v)};
}
// |a| ≥ |b| (ou a nul).
DD fast_two_sum(double a, double b) {
    const double s = a + b;
    return {s, b - (s - a)};
}
// Produit exact de Dekker (découpage 26 + 27 bits), sans FMA.
DD two_prod(double a, double b) {
    constexpr double split = 134217729.0;  // 2^27 + 1
    const double p = a*b;
    const double ta = split*a, ah = ta - (ta - a), al = a - ah;
    const double tb = split*b, bh = tb - (tb - b), bl = b - bh;
    return {p, ((ah*bh - p) + ah*bl + al*bh) + al*bl};
}
// Addition précise (erreur relative ≤ 3·2^-106, Joldes, Muller, Popescu 2017).
DD operator+(const DD &a, const DD &b) {
    DD s = two_sum(a.hi, b.hi);
    const DD t = two_sum(a.lo, b.lo);
    s = fast_two_sum(s.hi, s.lo + t.hi);
    return fast_two_sum(s.hi, s.lo + t.lo);
}
DD operator-(const DD &a) { return {-a.hi, -a.lo}; }
DD operator-(const DD &a, const DD &b) { return a + (-b); }
DD operator*(const DD &a, const DD &b) {
    const DD p = two_prod(a.hi, b.hi);
    return fast_two_sum(p.hi, p.lo + (a.hi*b.lo + a.lo*b.hi));
}
DD operator*(const DD &a, double b) {
    const DD p = two_prod(a.hi, b);
    return fast_two_sum(p.hi, p.lo + a.lo*b);
}
DD operator/(const DD &a, double b) {
    const double q = a.hi/b;
    const DD p = two_prod(q, b);
    return fast_two_sum(q, (((a.hi - p.hi) - p.lo) + a.lo)/b);
}
DD operator/(const DD &a, const DD &b) {
    const double q1 = a.hi/b.hi;
    DD r = a - b*q1;
    const double q2 = r.hi/b.hi;
    r = r - b*q2;
    const double q3 = r.hi/b.hi;
    return fast_two_sum(q1, q2) + DD{q3, 0};
}
bool operator<(const DD &a, const DD &b) { return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo); }
DD scaled(const DD &a, int exponent) { return {std::ldexp(a.hi, exponent), std::ldexp(a.lo, exponent)}; }
DD threshold(int k) { return {pq_threshold(k), pq_threshold_low(k)}; }
}

DD exp(const DD &a) {
    if (a.hi < -745) return {};
    if (a.hi > 709) return {INFINITY, 0};
    // a = n·ln 2 + r, |r| ≤ 0,347 ; e^r = (e^(r/32))^32.
    const double n = std::nearbyint(a.hi/ln2.hi);
    const DD r = scaled(a - ln2*n, -5);
    // e^r - 1 = r·(1 + r/2·(1 + r/3·(… (1 + r/14)))), |r| ≤ 0,0109 : reste < 1e-41.
    DD t{1, 0};
    for (int k = 14; k >= 2; --k) t = DD{1, 0} + (r/double(k))*t;
    DD m = r*t;
    // (1 + m)² - 1 = m·(2 + m), cinq fois, sans perte relative sur m.
    for (int i = 0; i < 5; ++i) m = m*(DD{2, 0} + m);
    return scaled(DD{1, 0} + m, static_cast<int>(n));
}

DD log(const DD &x) {
    // Newton sur exp, y ← y + x·e^-y - 1 : convergence quadratique depuis std::log (deux pas).
    DD y{std::log(x.hi), 0};
    for (int i = 0; i < 2; ++i) y = y + (x*exp(-y) - DD{1, 0});
    return y;
}

DD eotf(const DD &x, double &kappa) {
    kappa = 0;
    if (!(x.hi > 0)) return {};
    const DD p = exp(log(x)*32.0/2523.0);                 // x^(1/m2)
    const DD n = p - DD{c1, 0}, d = DD{c2, 0} - p*c3;
    if (!(d.hi > 0)) throw std::runtime_error("Signal PQ hors domaine");
    if (!(n.hi > 0)) return {};
    kappa = p.hi/m1m2*(1/n.hi + c3/d.hi);
    return exp(log(n/d)*8192.0/1305.0);                  // (n/d)^(1/m1)
}

double eotf_relative_error(double value, double kappa, double unit) {
    int exponent = 0;
    std::frexp(value, &exponent);
    return (800*kappa + 60 + 4.2*(std::abs(exponent) + 1))*unit;
}

unsigned classify(const Mapping &m, const ExactPixel &px, unsigned mask, std::array<uint16_t, 3> &codes,
                  std::array<DD, 3> *linear) {
    // Signaux BL exacts bornés à [0, 1].
    DD sig[3];
    sig[0] = px.luma <= 0 ? DD{} : px.luma >= 1023 ? DD{1, 0} : DD{double(px.luma), 0}/1023.0;
    for (int c = 1; c < 3; ++c) {
        const double num = double(px.chroma[c-1]), den = double(px.chroma_den[c-1])*1023;
        sig[c] = num <= 0 ? DD{} : num >= den ? DD{1, 0} : DD{num, 0}/den;
    }
    const DD terms[7] = {sig[0], sig[1], sig[2], sig[0]*sig[1], sig[0]*sig[2], sig[1]*sig[2],
                         sig[0]*sig[1]*sig[2]};
    // Mapping, NLQ et décalage : même structure de borne que le chemin double.
    DD v[3];
    double ev[3];
    for (int c = 0; c < 3; ++c) {
        const auto &curve = m.curves[c];
        const unsigned p = px.piece[c];
        DD s;
        if (curve.method[p] == AV_DOVI_MAPPING_POLYNOMIAL) {
            const auto &a = curve.poly[p];
            s = (sig[c]*a[2] + DD{a[1], 0})*sig[c] + DD{a[0], 0};
        } else {
            s = {curve.constant[p], 0};
            DD powers[7];
            std::copy(terms, terms + 7, powers);
            for (unsigned k = 0; k < curve.order[p]; ++k)
                for (int j = 0; j < 7; ++j) {
                    s = s + powers[j]*curve.mmr[p][k][j];
                    powers[j] = powers[j]*terms[j];
                }
        }
        const DD low = DD{double(curve.codes[0]), 0}/m.bl_max;
        const DD high = DD{double(curve.codes[curve.count-1]), 0}/m.bl_max;
        if (s < low) s = low;
        else if (high < s) s = high;
        ev[c] = (curve.weight[p] + 1)*u;
        if (m.residual && px.residual[c] != 0) {
            const DD d = DD{double(std::llabs(px.residual[c])), 0}/(double(px.residual_den[c])*1023);
            const DD q = d*m.slope[c] + DD{m.threshold[c], 0};
            s = px.residual[c] > 0 ? s + q : s - q;
            ev[c] += 5.3*u*(d.hi*std::abs(m.slope[c]) + std::abs(m.threshold[c]));
        }
        v[c] = s - DD{m.nonlinear_offset[c], 0};
        ev[c] += 2.02*u*(std::abs(v[c].hi) + std::abs(m.nonlinear_offset[c]));
    }
    // Signaux LMS' et EOTF : encadrement propagé par le conditionnement.
    DD L[3];
    double eL[3];
    for (int i = 0; i < 3; ++i) {
        const DD x = v[0]*m.nonlinear[i][0] + v[1]*m.nonlinear[i][1] + v[2]*m.nonlinear[i][2];
        double e = 0, a = 0;
        for (int j = 0; j < 3; ++j) {
            e += std::abs(m.nonlinear[i][j])*ev[j];
            a += std::abs(m.nonlinear[i][j]*v[j].hi);
        }
        e = 1.01*(e + 3.01*u*a);
        double kappa;
        L[i] = eotf(x, kappa);
        if (x.hi + e <= PQ_ZERO_LOW) { L[i] = {}; eL[i] = 0; }
        else if (x.hi - e > PQ_ZERO_HIGH && kappa*e <= 1e-6*x.hi && e <= 1e-6*x.hi)
            eL[i] = L[i].hi*(eotf_relative_error(L[i].hi, kappa, u) + 1.01*kappa*e/x.hi) + 1e-300;
        else eL[i] = INFINITY;
    }
    // Valeurs linéaires RGB (matrice exacte : produits de Dekker), encadrement puis seuils.
    const Mat &A = lms_to_rgb_matrix();
    unsigned undecided = 0;
    for (int c = 0; c < 3; ++c) {
        if (!(mask >> c & 1)) continue;
        DD R{};
        double e = 0;
        for (int k = 0; k < 3; ++k) {
            DD M{};
            double magnitude = 0;
            for (int j = 0; j < 3; ++j) {
                M = M + two_prod(A[c][j], m.linear[j][k]);
                magnitude += std::abs(A[c][j]*m.linear[j][k]);
            }
            const double dM = 3.01*u*magnitude;
            R = R + M*L[k];
            e += (std::abs(M.hi)*(1 + 4*u) + dM)*eL[k] + (dM + 3.01*u*std::abs(M.hi))*L[k].hi;
        }
        if (!std::isfinite(R.hi)) throw std::runtime_error("Pixel reconstruit non fini");
        if (linear) (*linear)[c] = R;
        e = e*(1 + 1e-6) + 0x1p-98*std::abs(R.hi) + 1e-300;
        // Code candidat : seuils hauts par dichotomie, puis seuils double-double.
        int lo = 0, hi = PQ_THRESHOLDS;
        while (lo < hi) {
            const int mid = (lo + hi)/2;
            if (pq_threshold(mid) < R.hi) lo = mid + 1;
            else hi = mid;
        }
        int k = lo;
        while (k > 0 && R < threshold(k-1)) --k;
        while (k < PQ_THRESHOLDS && !(R < threshold(k))) ++k;
        // Marge : erreur de la table (2^-105) et soustraction double-double.
        auto separated = [&](const DD &difference, const DD &t) {
            return difference.hi*(1 - 0x1p-52) > e + 0x1p-98*(std::abs(R.hi) + t.hi);
        };
        const bool low_ok = k == 0 || separated(R - threshold(k-1), threshold(k-1));
        const bool high_ok = k == PQ_THRESHOLDS || separated(threshold(k) - R, threshold(k));
        if (!(low_ok && high_ok)) undecided |= 1u << c;
        codes[c] = static_cast<uint16_t>(k);
    }
    return undecided;
}
}
