// SPDX-License-Identifier: LGPL-2.1-or-later
// Rendu certifié en virgule fixe multi-mots (voir fixed_point.hpp) : référence du shader GPU.
#include "fixed_point.hpp"
#include "fixed_abi.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>

namespace muxiveo::fel::fixed {
namespace {
using certified::DD;

// Arrondi au plus proche de v·2^bits (double-double) en entier 128 bits.
i128 to_fixed(const DD &v, int bits) {
    const long double h = std::ldexp((long double)v.hi, bits), l = std::ldexp((long double)v.lo, bits);
    const long double f = std::floor(h);
    return (i128)f + (i128)std::llround((double)((h - f) + l));
}
// 1/k! en entier (arrondi) à l'échelle 2^bits.
uint64_t inverse_factorial(int k, int bits) {
    u128 f = 1;
    for (int i = 2; i <= k; ++i) f *= i;
    return (uint64_t)((((u128)1 << bits) + f/2)/f);
}
uint64_t mulhi(uint64_t a, uint64_t b) { return (uint64_t)(((u128)a*b) >> 64); }
// S2.62 × S63 → S2.62, tronqué vers zéro (magnitudes).
int64_t smul(int64_t a, int64_t b) {
    const u128 m = (u128)(uint64_t)(a < 0 ? -a : a)*(uint64_t)(b < 0 ? -b : b);
    const int64_t r = (int64_t)(m >> 63);
    return (a < 0) != (b < 0) ? -r : r;
}
int msb(u128 x) { return x >> 64 ? 127 - __builtin_clzll((uint64_t)(x >> 64)) : 63 - __builtin_clzll((uint64_t)x); }
// floor(a·2^64/b), 0 ≤ a ≤ b (≥ 1 : 2^64 - 1).
uint64_t ratio64(uint64_t a, uint64_t b) { return a >= b ? ~(uint64_t)0 : (uint64_t)(((u128)a << 64)/b); }

// ln(v) pour v > 0 à l'échelle 2^-fraction, en S5.58.
int64_t ln_fixed(u128 v, int fraction) {
    const Tables &t = tables();
    const int top = msb(v), e = top - fraction;
    const uint64_t m = top >= 63 ? (uint64_t)(v >> (top - 63)) : (uint64_t)(v << (63 - top));   // Q1.63
    const int j = (int)((m >> 57) & 63);
    const uint64_t c = ((uint64_t)(64 + j) << 57) | ((uint64_t)1 << 56);
    const int64_t d = (int64_t)(m - c);                                    // |d| ≤ 2^56
    const uint64_t rd = mulhi((uint64_t)(d < 0 ? -d : d), t.inv_c[j]);
    const int64_t r = d < 0 ? -(int64_t)rd : (int64_t)rd;                  // S63, |r| < 2^-7
    int64_t acc = t.ln_coef[8];
    for (int k = 7; k >= 1; --k) acc = t.ln_coef[k] + smul(acc, r);
    const int64_t series = smul(acc, r);                                   // S2.62
    const int64_t eln2 = (int64_t)(((i128)e*(i128)t.ln2_q64) >> 6);
    return eln2 + (t.ln_c[j] >> 5) + (series >> 4);
}

// e^y, y ≤ 5 en S8.55 : mantisse Q1.63 et exposant (valeur = mant·2^(n-63)).
uint64_t exp_fixed(int64_t y, int &n) {
    const Tables &t = tables();
    // K = floor(y·64/ln2) (constante Q7.56) ; r = y - K·ln2/64 exact, 0 ≤ r < ln2/64 à 2^-63 près.
    const int64_t K = (int64_t)(((i128)y*(i128)t.c64_ln2_q56) >> 111);
    const u128 ln2_64 = (u128)t.ln2_64_q90[1] << 64 | t.ln2_64_q90[0];
    const i128 r90 = ((i128)y << 35) - (i128)K*(i128)ln2_64;
    const int64_t r = (int64_t)(r90 >> 27);                                // S63
    int64_t acc = t.exp_coef[8];
    for (int k = 7; k >= 0; --k) acc = t.exp_coef[k] + smul(acc, r);      // e^r en S2.62
    const int64_t q = K >= 0 ? K/64 : -((-K + 63)/64);
    const int j = (int)(K - 64*q);
    const u128 product = (u128)t.pow2[j]*(uint64_t)acc;                    // Q3.125
    int shift = 62;
    while (product >> (shift + 64)) ++shift;
    n = (int)q + (shift - 62);
    return (uint64_t)(product >> shift);
}
}

const Tables &tables() {
    static const Tables t = [] {
        Tables v{};
        const DD ln2 = certified::log(DD{2, 0});
        for (int j = 0; j < 64; ++j) {
            const DD c{1 + (2*j + 1)/128.0, 0}, l = certified::log(c);
            v.ln_c[j] = (int64_t)to_fixed(l, 63);
            v.inv_c[j] = (uint64_t)to_fixed(certified::exp(DD{-l.hi, -l.lo}), 64);
            const double hi = ln2.hi*j/64.0, err = std::fma(ln2.hi, j/64.0, -hi);
            v.pow2[j] = (uint64_t)to_fixed(certified::exp(DD{hi, err + ln2.lo*j/64.0}), 63);
        }
        for (int j = 0; j < 16; ++j) {
            const DD e = certified::exp(DD{j/64.0, 0});
            v.expm1[j] = (uint64_t)to_fixed(DD{e.hi - 1, e.lo}, 64);
        }
        for (int k = 1; k <= 8; ++k) v.ln_coef[k] = (k & 1 ? 1 : -1)*(int64_t)((((u128)1 << 62) + k/2)/k);
        for (int k = 0; k <= 8; ++k) v.exp_coef[k] = (int64_t)inverse_factorial(k, 62);
        for (int k = 2; k <= 8; ++k) v.expm1_coef[k] = inverse_factorial(k, 64);
        v.ln2_q64 = (uint64_t)to_fixed(ln2, 64);
        v.c32_2523_q64 = (uint64_t)((((u128)32 << 64) + 2523/2)/2523);
        v.c8192_1305_q61 = (uint64_t)((((u128)8192 << 61) + 1305/2)/1305);
        v.c64_ln2_q56 = (uint64_t)to_fixed(certified::exp(DD{-certified::log(ln2).hi, -certified::log(ln2).lo}), 62);
        // ln x0 = m2·ln c1 : ln c1 en double-double, produit par 2523/32 en entier.
        const DD lc1 = certified::log(DD{3424.0/4096, 0});
        v.lnx0_s58 = (int64_t)((to_fixed(lc1, 90)*2523/32) >> 32);
        const u128 l = (u128)to_fixed(ln2, 84);
        v.ln2_64_q90[0] = (uint64_t)l; v.ln2_64_q90[1] = (uint64_t)(l >> 64);
        return v;
    }();
    return t;
}

bool prepare(const Mapping &mapping, const AVDOVIMetadata &metadata, Params &p) {
    const auto &h = *av_dovi_get_header(&metadata);
    const auto &m = *av_dovi_get_mapping(&metadata);
    p = Params{};
    if (h.bl_bit_depth != 10 || h.el_bit_depth != 10 || h.coef_log2_denom > 32) return false;
    p.d = (int)h.coef_log2_denom;
    p.residual = mapping.residual;
    const double unit = std::ldexp(1.0, -p.d);
    auto fits = [](long long v) { return v >= INT32_MIN && v <= INT32_MAX; };
    for (int c = 0; c < 3; ++c) {
        const auto &src = m.curves[c];
        auto &dst = p.curves[c];
        dst.count = src.num_pivots;
        for (unsigned k = 0; k < dst.count; ++k) dst.codes[k] = src.pivots[k];
        // Bornes code/1023 à l'échelle 2^-(64+d), arrondies vers le bas.
        dst.low = (i128)(((u128)dst.codes[0] << (64 + p.d))/1023);
        dst.high = (i128)(((u128)dst.codes[dst.count - 1] << (64 + p.d))/1023);
        for (unsigned k = 0; k + 1 < dst.count; ++k) {
            double magnitude = 0;
            dst.method[k] = src.mapping_idc[k];
            if (src.mapping_idc[k] == AV_DOVI_MAPPING_POLYNOMIAL) {
                for (unsigned j = 0; j <= src.poly_order[k]; ++j) {
                    if (!fits(src.poly_coef[k][j])) return false;
                    dst.coef[k][j] = (int32_t)src.poly_coef[k][j];
                    if (j) magnitude += std::abs((double)src.poly_coef[k][j]);
                }
            } else {
                dst.order[k] = src.mmr_order[k];
                if (!fits(src.mmr_constant[k])) return false;
                dst.coef[k][0] = (int32_t)src.mmr_constant[k];
                for (int o = 0; o < dst.order[k]; ++o)
                    for (int j = 0; j < 7; ++j) {
                        if (!fits(src.mmr_coef[k][o][j])) return false;
                        dst.coef[k][1 + 7*o + j] = (int32_t)src.mmr_coef[k][o][j];
                        magnitude += std::abs((double)src.mmr_coef[k][o][j]);
                    }
            }
            // Monômes Q0.64 tronqués : erreur < 17·2^-64 (produits et puissances d'ordre ≤ 3) ;
            // bornage et décalage tronqués : 2 unités 2^-(64+d).
            dst.error[k] = 20*std::ldexp(magnitude*unit, -64) + std::ldexp(2.0, -(64 + p.d));
        }
        p.slope[c] = (int64_t)m.nlq[c].linear_deadzone_slope;
        p.threshold2[c] = 2*(int64_t)m.nlq[c].linear_deadzone_threshold - p.slope[c];
        const long double off = std::ldexp((long double)mapping.nonlinear_offset[c], 64 + p.d);
        p.offset[c] = (i128)std::floor(off);
    }
    // Matrice Y'CbCr→LMS' : n·2^-q avec n entier sur 32 bits.
    for (p.q = 0; p.q <= 30; ++p.q) {
        bool exact = true;
        for (int i = 0; i < 9 && exact; ++i) {
            const double v = std::ldexp(mapping.nonlinear[i/3][i%3], p.q);
            exact = v == std::floor(v) && std::abs(v) < 2147483648.0;
        }
        if (exact) break;
    }
    if (p.q > 30) return false;
    for (int i = 0; i < 9; ++i) p.n[i] = (int32_t)std::ldexp(mapping.nonlinear[i/3][i%3], p.q);
    // Matrice LMS→RGB : produit en double (comme le CPU), erreur 3,01·2^-53·Σ|A·B|, puis S2.61.
    const Mat &a = lms_to_rgb_matrix();
    for (int c = 0; c < 3; ++c)
        for (int k = 0; k < 3; ++k) {
            double sum = 0, magnitude = 0;
            for (int j = 0; j < 3; ++j) {
                sum += a[c][j]*mapping.linear[j][k];
                magnitude += std::abs(a[c][j]*mapping.linear[j][k]);
            }
            p.m[3*c + k] = (int64_t)std::llround(std::ldexp(sum, 61));
            p.m_error[3*c + k] = 3.01*std::ldexp(magnitude, -53) + std::ldexp(1.0, -62);
        }
    return p.ok = true;
}

bool eotf(u128 x, int fraction, uint64_t &mant, int &n, double &dl, double *kappa) {
    const Tables &t = tables();
    mant = 0; n = 0; dl = 0;
    if (kappa) *kappa = 0;
    if (!x) return true;
    const int64_t dls = ln_fixed(x, fraction) - t.lnx0_s58;
    dl = std::ldexp((double)dls, -58);
    if (dls <= 0) return true;
    const uint64_t delta = (uint64_t)(((u128)(uint64_t)dls*t.c32_2523_q64) >> 59);   // S63
    const int j = (int)(delta >> 57);
    if (j > 15) return false;
    const uint64_t r = (delta - ((uint64_t)j << 57)) << 1;                // Q0.64, r < 1/64
    uint64_t b = t.expm1_coef[8];
    for (int k = 7; k >= 2; --k) b = t.expm1_coef[k] + mulhi(b, r);
    const uint64_t em = r + mulhi(mulhi(r, r), b);
    const uint64_t expm1 = em + mulhi(t.expm1[j], em) + t.expm1[j];
    const uint64_t num = (uint64_t)(((u128)expm1*107) >> 7);               // c1·expm1, Q0.64
    // Cadre : num < 43/256 (x ≲ 1,33), soit g < 2 ; au-delà (vers le pôle), reprise CPU.
    if (num >= ((uint64_t)43 << 56)) return false;
    const uint64_t den = ((uint64_t)6615 << 51) - (uint64_t)(((u128)num*299) >> 6);   // Q2.62
    // g = num/den en Q1.63 : estimation flottante, corrections par le résidu exact, vérification.
    uint64_t g = (uint64_t)std::min(18446744073709551615.0, (double)((float)num/((float)den*4.0f))*9223372036854775808.0);
    i128 res = 0;
    for (int it = 0; it < 3; ++it) {
        res = ((i128)(u128)num << 61) - (i128)((u128)g*den);
        g = (uint64_t)((i128)g + (i128)std::llround((double)((float)res/(float)den)));
    }
    // Correction exacte : 0 ≤ résidu < den, soit g = floor(num·2^61/den).
    for (int it = 0; it < 2; ++it) {
        res = ((i128)(u128)num << 61) - (i128)((u128)g*den);
        if (res < 0) --g;
        else if (res >= (i128)den) ++g;
    }
    res = ((i128)(u128)num << 61) - (i128)((u128)g*den);
    if (res < 0 || res >= (i128)den) return false;
    // κ = p/(m1·m2)·(1/num + c3/den), p = c1 + num : sans annulation.
    if (kappa) {
        const double nf = std::ldexp((double)num, -64), df = std::ldexp((double)den, -62);
        *kappa = (0.8359375 + nf)/(6585030.0/524288)*(1/nf + 18.6875/df);
    }
    if (!g) return true;
    const int64_t lg = ln_fixed(g, 63);
    const int64_t y = (int64_t)(((i128)lg*(i128)t.c8192_1305_q61) >> 64);  // S8.55
    mant = exp_fixed(y, n);
    return true;
}

unsigned pixel(const Params &p, const certified::ExactPixel &in, std::array<uint16_t, 3> &codes,
               i128 (&linear)[3], double (&bound)[3]) {
    // Signaux BL en Q0.64 (arrondis vers le bas ; 1 → 2^64 - 1).
    uint64_t sig[3];
    sig[0] = ratio64((uint64_t)std::clamp(in.luma, 0, 1023), 1023);
    for (int c = 1; c < 3; ++c) {
        const int64_t den = 1023*in.chroma_den[c-1];
        sig[c] = ratio64((uint64_t)std::clamp<int64_t>(in.chroma[c-1], 0, den), (uint64_t)den);
    }
    uint64_t terms[3][7];
    terms[0][0] = sig[0]; terms[0][1] = sig[1]; terms[0][2] = sig[2];
    terms[0][3] = mulhi(sig[0], sig[1]); terms[0][4] = mulhi(sig[0], sig[2]); terms[0][5] = mulhi(sig[1], sig[2]);
    terms[0][6] = mulhi(terms[0][3], sig[2]);
    for (int j = 0; j < 7; ++j) { terms[1][j] = mulhi(terms[0][j], terms[0][j]); terms[2][j] = mulhi(terms[1][j], terms[0][j]); }
    const int scale = 64 + p.d;
    i128 v[3];
    double ev[3];
    for (int c = 0; c < 3; ++c) {
        const auto &curve = p.curves[c];
        const unsigned k = in.piece[c];
        i128 s;
        if (curve.method[k] == AV_DOVI_MAPPING_POLYNOMIAL)
            s = ((i128)curve.coef[k][0] << 64) + (i128)curve.coef[k][1]*(i128)sig[c] +
                (i128)curve.coef[k][2]*(i128)mulhi(sig[c], sig[c]);
        else {
            s = (i128)curve.coef[k][0] << 64;
            for (int o = 0; o < curve.order[k]; ++o)
                for (int j = 0; j < 7; ++j) s += (i128)curve.coef[k][1 + 7*o + j]*(i128)terms[o][j];
        }
        // Bornage exact aux pivots : s·1023 face à code·2^(64+d).
        if (s*1023 < ((i128)curve.codes[0] << scale)) s = curve.low;
        else if (s*1023 > ((i128)curve.codes[curve.count - 1] << scale)) s = curve.high;
        ev[c] = curve.error[k];
        if (p.residual && in.residual[c] != 0) {
            const u128 numerator = (u128)std::llabs(in.residual[c])*(u128)p.slope[c];
            const i128 q = (i128)((numerator << 64)/(u128)in.residual_den[c]) + ((i128)p.threshold2[c] << 63);
            s = in.residual[c] > 0 ? s + q : s - q;
            ev[c] += std::ldexp(1.0, -scale);
        }
        v[c] = s - p.offset[c];
    }
    // Signaux LMS' à l'échelle 2^-(64+d+q), EOTF, encadrement.
    const int fraction = scale + p.q;
    uint64_t mant[3];
    int expo[3];
    double L[3], rel[3];
    unsigned uncertain = 0;
    for (int i = 0; i < 3; ++i) {
        const i128 x = (i128)p.n[3*i]*v[0] + (i128)p.n[3*i+1]*v[1] + (i128)p.n[3*i+2]*v[2];
        double ex = std::ldexp(1.0, -fraction);
        for (int j = 0; j < 3; ++j) ex += std::ldexp(std::abs((double)p.n[3*i+j]), -p.q)*ev[j];
        ex *= 1.01;
        const double xv = std::ldexp((double)x, -fraction);
        double dl = 0;
        mant[i] = 0; expo[i] = 0; L[i] = 0; rel[i] = 0;
        if (x <= 0 || xv + ex <= certified::PQ_ZERO_LOW) {
            if (!(xv + ex <= certified::PQ_ZERO_LOW)) uncertain = 7;   // zéro de l'EOTF à cheval
            continue;
        }
        double kappa = 0;
        if (!eotf((u128)x, fraction, mant[i], expo[i], dl, &kappa)) { uncertain = 7; continue; }
        if (!mant[i]) { if (!(xv + ex <= certified::PQ_ZERO_LOW)) uncertain = 7; continue; }
        L[i] = std::ldexp((double)mant[i], expo[i] - 63);
        // Conditionnement κ (num et den exacts, marge 1,01) et erreur de l'EOTF en virgule fixe.
        if (!(xv - ex > certified::PQ_ZERO_HIGH && kappa*ex <= 1e-6*xv && ex <= 1e-6*xv && dl > 0)) { uncertain = 7; continue; }
        rel[i] = std::ldexp(1.0, -50)*(1 + 1/dl) + 1.01*kappa*ex/xv;
    }
    // Matrice LMS→RGB en Q0.96 : L·2^96 = mant·2^(n+33).
    u128 Lq[3];
    for (int k = 0; k < 3; ++k) {
        const int sh = expo[k] + 33;
        Lq[k] = !mant[k] ? 0 : sh >= 0 ? (u128)mant[k] << sh : sh > -64 ? (u128)(mant[k] >> -sh) : 0;
    }
    for (int c = 0; c < 3; ++c) {
        i128 R = 0;
        double e = 3*std::ldexp(1.0, -96);
        for (int k = 0; k < 3; ++k) {
            const int64_t mk = p.m[3*c + k];
            const u128 am = (u128)(uint64_t)(mk < 0 ? -mk : mk);
            const u128 lo = (uint64_t)Lq[k], hi = (uint64_t)(Lq[k] >> 64);
            const i128 prod = (i128)((am*hi) << 3) + (i128)((am*lo) >> 61);   // |M|·L / 2^61, Q0.96
            R += mk < 0 ? -prod : prod;
            const double am_v = std::ldexp((double)am, -61);
            e += ((am_v + p.m_error[3*c + k])*rel[k] + p.m_error[3*c + k])*L[k]*(1 + 1e-12) + std::ldexp(2.0, -96);
        }
        linear[c] = R;
        bound[c] = e*(1 + 1e-9);
        // Classement : code candidat par les seuils doubles, puis comparaisons exactes en Q0.96.
        const double Rd = std::ldexp((double)R, -96);
        int lo = 0, hi = PQ_THRESHOLDS;
        while (lo < hi) { const int mid = (lo + hi)/2; if (pq_threshold(mid) < Rd) lo = mid + 1; else hi = mid; }
        const int k = lo;
        // Seuil exact T dans ]T_bas, suivant(T_bas)[ : borne basse comparée à suivant(T_bas) (R n'est pas
        // un double), borne haute à T_bas.
        auto T = [](int i) { return (i128)(u128)std::ldexp(pq_threshold(i), 96); };
        auto T_up = [](int i) { return (i128)(u128)std::ldexp(std::nextafter(pq_threshold(i), 2.0), 96); };
        const i128 eq = (i128)(u128)std::ceil(std::ldexp(bound[c], 96)) + 1;
        const bool certain = (k == 0 || R - eq >= T_up(k - 1)) && (k == PQ_THRESHOLDS || R + eq <= T(k));
        codes[c] = (uint16_t)k;
        if (!certain || (uncertain & 7) == 7) uncertain |= 1u << c;
    }
    return uncertain;
}
}

// Interface C du backend GPU (voir fixed_abi.h).
extern "C" int fel_fixed_tables(uint32_t *words, int capacity) {
    using namespace muxiveo::fel::fixed;
    if (capacity < FEL_FIXED_TABLE_WORDS) return 0;
    const Tables &t = tables();
    auto put = [&](int at, uint64_t v) { words[2*at] = (uint32_t)v; words[2*at + 1] = (uint32_t)(v >> 32); };
    for (int j = 0; j < 64; ++j) {
        put(FXT_LNC + j, (uint64_t)t.ln_c[j]); put(FXT_INVC + j, t.inv_c[j]); put(FXT_POW2 + j, t.pow2[j]);
    }
    for (int j = 0; j < 16; ++j) put(FXT_EXPM1 + j, t.expm1[j]);
    for (int k = 0; k < 9; ++k) {
        put(FXT_LNK + k, (uint64_t)t.ln_coef[k]); put(FXT_EXPK + k, (uint64_t)t.exp_coef[k]); put(FXT_EM1K + k, t.expm1_coef[k]);
    }
    const uint64_t scalars[7] = {t.ln2_q64, t.c32_2523_q64, t.c8192_1305_q61, t.c64_ln2_q56, (uint64_t)t.lnx0_s58,
                                 t.ln2_64_q90[0], t.ln2_64_q90[1]};
    for (int i = 0; i < 7; ++i) put(FXT_CONST + i, scalars[i]);
    return FEL_FIXED_TABLE_WORDS;
}

extern "C" int fel_fixed_params(const AVDOVIMetadata *metadata, uint32_t *words, int capacity) {
    using namespace muxiveo::fel::fixed;
    if (capacity < FEL_FIXED_WORDS) return 0;
    try {
        const muxiveo::fel::Mapping mapping(*metadata);
        Params p;
        if (!prepare(mapping, *metadata, p)) return 0;
        // Cadre du shader : |n| < 2^20 et d ≤ 28 (produits n·v sur 128 bits).
        if (p.d > 28) return 0;
        for (int i = 0; i < 9; ++i) if (std::abs(p.n[i]) >= (1 << 20)) return 0;
        for (int c = 0; c < 3; ++c) if (p.slope[c] < 0 || p.slope[c] > 0xffffffffLL) return 0;   // pente sur 32 bits
        std::memset(words, 0, sizeof(uint32_t)*FEL_FIXED_WORDS);
        auto put64 = [&](int at, uint64_t v) { words[at] = (uint32_t)v; words[at + 1] = (uint32_t)(v >> 32); };
        auto put128 = [&](int at, i128 v) { put64(at, (uint64_t)(u128)v); put64(at + 2, (uint64_t)((u128)v >> 64)); };
        auto putf = [&](int at, double v) { const float f = (float)(v*(1 + 1e-6)); std::memcpy(&words[at], &f, 4); };
        words[0] = (uint32_t)p.d; words[1] = (uint32_t)p.q; words[2] = p.residual;
        for (int c = 0; c < 3; ++c) {
            const auto &curve = p.curves[c];
            const int b = FEL_FIXED_CURVE0 + c*FEL_FIXED_CURVE;
            words[b] = curve.count;
            for (int k = 0; k < 9; ++k) words[b + 1 + k] = (uint32_t)curve.codes[k];
            for (int k = 0; k < 8; ++k) { words[b + 10 + k] = (uint32_t)curve.method[k]; words[b + 18 + k] = (uint32_t)curve.order[k]; }
            for (int k = 0; k < 8; ++k) for (int i = 0; i < 24; ++i) words[b + 26 + 24*k + i] = (uint32_t)curve.coef[k][i];
            put128(b + 218, curve.low); put128(b + 222, curve.high);
            for (int k = 0; k < 8; ++k) putf(b + 226 + k, curve.error[k]);
        }
        const int g = FEL_FIXED_GLOBAL;
        for (int c = 0; c < 3; ++c) {
            put64(g + 2*c, (uint64_t)p.slope[c]); put64(g + 6 + 2*c, (uint64_t)p.threshold2[c]);
            put128(g + 12 + 4*c, p.offset[c]);
        }
        for (int i = 0; i < 9; ++i) {
            words[g + 24 + i] = (uint32_t)p.n[i];
            put64(g + 33 + 2*i, (uint64_t)p.m[i]);
            putf(g + 51 + i, p.m_error[i]);
        }
        return FEL_FIXED_WORDS;
    } catch (const std::exception &) {
        return 0;
    }
}
