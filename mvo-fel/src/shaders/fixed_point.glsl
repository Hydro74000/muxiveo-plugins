// SPDX-License-Identifier: LGPL-2.1-or-later
// Virgule fixe multi-mots du rendu certifié (lot A2') : traduction opération par opération de
// src/fixed_point.cpp (mêmes formats, mêmes arrondis). uvec2 = 64 bits (x poids faible),
// uvec4 = 128 bits, complément à deux pour les valeurs signées. Tables (tableau uniforme fx_t, deux
// valeurs 64 bits par uvec4, décalages FXT_* de fixed_abi.h) et paramètres de l'image (fel_fp)
// déclarés avant ce fichier : aucun tableau constant indexé dynamiquement (mémoire locale).
uvec2 fxt(int i) { uvec4 v = fx_t[i >> 1]; return (i & 1) == 0 ? v.xy : v.zw; }

// ---- 64 bits
uvec2 x2add(uvec2 a, uvec2 b) { uint c; uint l = uaddCarry(a.x, b.x, c); return uvec2(l, a.y + b.y + c); }
uvec2 x2sub(uvec2 a, uvec2 b) { uint c; uint l = usubBorrow(a.x, b.x, c); return uvec2(l, a.y - b.y - c); }
uvec2 x2neg(uvec2 a) { return x2sub(uvec2(0u), a); }
bool x2sign(uvec2 a) { return int(a.y) < 0; }
uvec2 x2abs(uvec2 a) { return x2sign(a) ? x2neg(a) : a; }
bool x2lt(uvec2 a, uvec2 b) { return a.y < b.y || (a.y == b.y && a.x < b.x); }   // non signé
uvec2 x2sar(uvec2 a, int n) { return uvec2((a.x >> n) | (a.y << (32 - n)), uint(int(a.y) >> n)); }   // 0 < n < 32
float x2float(uvec2 a) { return float(a.y)*4294967296.0 + float(a.x); }          // non signé, approché
uvec2 x2from(float v) {                                                           // v ≥ 0, approché
    v = max(v, 0.0);
    float h = min(floor(v*2.3283064e-10), 4294967295.0);
    return uvec2(uint(clamp(v - h*4294967296.0, 0.0, 4294967295.0)), uint(h));
}

// ---- 128 bits
uvec4 x4add(uvec4 a, uvec4 b) {
    uint k0, k1, k2; uvec4 r;
    r.x = uaddCarry(a.x, b.x, k0);
    uint t = uaddCarry(a.y, b.y, k1); r.y = uaddCarry(t, k0, k2); k1 += k2;
    t = uaddCarry(a.z, b.z, k0); r.z = uaddCarry(t, k1, k2); k0 += k2;
    r.w = a.w + b.w + k0;
    return r;
}
uvec4 x4neg(uvec4 a) { return x4add(~a, uvec4(1u, 0u, 0u, 0u)); }
uvec4 x4sub(uvec4 a, uvec4 b) { return x4add(a, x4neg(b)); }
bool x4sign(uvec4 a) { return int(a.w) < 0; }
uvec4 x4abs(uvec4 a) { return x4sign(a) ? x4neg(a) : a; }
bool x4zero(uvec4 a) { return (a.x | a.y | a.z | a.w) == 0u; }
// Comparaison signée : complément à deux de même signe comparé comme non signé (sans indexation
// dynamique des composantes : registres seulement).
bool x4lt(uvec4 a, uvec4 b) {
    bool sa = x4sign(a), sb = x4sign(b);
    if (sa != sb) return sa;
    if (a.w != b.w) return a.w < b.w;
    if (a.z != b.z) return a.z < b.z;
    if (a.y != b.y) return a.y < b.y;
    return a.x < b.x;
}
uvec4 x4ext(uvec2 a) { uint s = x2sign(a) ? 0xffffffffu : 0u; return uvec4(a, s, s); }
uvec4 x4shr(uvec4 a, int n) {                                                     // logique, 0 ≤ n < 128
    if (n >= 64) { a = uvec4(a.zw, 0u, 0u); n -= 64; }
    if (n >= 32) { a = uvec4(a.yzw, 0u); n -= 32; }
    if (n == 0) return a;
    return uvec4((a.x >> n) | (a.y << (32 - n)), (a.y >> n) | (a.z << (32 - n)), (a.z >> n) | (a.w << (32 - n)), a.w >> n);
}
uvec4 x4sar(uvec4 a, int n) {                                                     // arithmétique (plancher)
    if (!x4sign(a)) return x4shr(a, n);
    return x4neg(x4add(x4shr(x4sub(x4neg(a), uvec4(1u, 0u, 0u, 0u)), n), uvec4(1u, 0u, 0u, 0u)));
}
uvec4 x4shl(uvec4 a, int n) {                                                     // 0 ≤ n < 128
    if (n >= 64) { a = uvec4(0u, 0u, a.xy); n -= 64; }
    if (n >= 32) { a = uvec4(0u, a.xyz); n -= 32; }
    if (n == 0) return a;
    return uvec4(a.x << n, (a.y << n) | (a.x >> (32 - n)), (a.z << n) | (a.y >> (32 - n)), (a.w << n) | (a.z >> (32 - n)));
}
int x4msb(uvec4 a) {
    return a.w != 0u ? 96 + findMSB(a.w) : a.z != 0u ? 64 + findMSB(a.z) : a.y != 0u ? 32 + findMSB(a.y) : findMSB(a.x);
}
float x4float(uvec4 a) {                                                          // signé, approché
    bool n = x4sign(a); uvec4 m = x4abs(a);
    float v = float(m.w)*7.9228163e28 + float(m.z)*1.8446744e19 + float(m.y)*4294967296.0 + float(m.x);
    return n ? -v : v;
}

// ---- produits
uvec4 x2mul(uvec2 a, uvec2 b) {                                                   // 64 × 64 → 128
    uint h00, l00, h01, l01, h10, l10, h11, l11;
    umulExtended(a.x, b.x, h00, l00); umulExtended(a.x, b.y, h01, l01);
    umulExtended(a.y, b.x, h10, l10); umulExtended(a.y, b.y, h11, l11);
    uint c1, c2; uint w1 = uaddCarry(h00, l01, c1); w1 = uaddCarry(w1, l10, c2);
    uint d1, d2, d3; uint w2 = uaddCarry(h01, h10, d1); w2 = uaddCarry(w2, l11, d2); w2 = uaddCarry(w2, c1 + c2, d3);
    return uvec4(l00, w1, w2, h11 + d1 + d2 + d3);
}
uvec2 x2hi(uvec2 a, uvec2 b) { return x2mul(a, b).zw; }                           // plancher(a·b / 2^64)
uvec4 x4mulu(uvec4 a, uint m) {                                                   // 128 × 32 → 128 (sans dépassement)
    uint h0, l0, h1, l1, h2, l2, h3, l3;
    umulExtended(a.x, m, h0, l0); umulExtended(a.y, m, h1, l1); umulExtended(a.z, m, h2, l2); umulExtended(a.w, m, h3, l3);
    uint k1, k2, k3; uint r1 = uaddCarry(l1, h0, k1); uint r2 = uaddCarry(l2, h1, k2); r2 = uaddCarry(r2, k1, k3);
    return uvec4(l0, r1, r2, l3 + h2 + k2 + k3);
}
uvec4 x4muls(uvec4 a, int m) {                                                    // signé × int32
    bool n = x4sign(a) != (m < 0);
    uvec4 p = x4mulu(x4abs(a), uint(abs(m)));
    return n ? x4neg(p) : p;
}
uvec4 x2muls(uvec2 t, int m) {                                                    // u64 × int32 → signé 128
    uint h0, l0, h1, l1; uint am = uint(abs(m));
    umulExtended(t.x, am, h0, l0); umulExtended(t.y, am, h1, l1);
    uint k; uint w1 = uaddCarry(h0, l1, k);
    uvec4 p = uvec4(l0, w1, h1 + k, 0u);
    return m < 0 ? x4neg(p) : p;
}
// S2.62 × S63 → S2.62, tronqué vers zéro (magnitudes).
uvec2 fx_smul(uvec2 a, uvec2 b) {
    bool n = x2sign(a) != x2sign(b);
    uvec2 r = x4shr(x2mul(x2abs(a), x2abs(b)), 63).xy;
    return n ? x2neg(r) : r;
}

// floor(N/D), N ≥ 0 sur 128 bits, D > 0 sur 64 bits, quotient < 2^64 : estimation flottante,
// trois corrections par le reste, deux corrections exactes ; ok faux si 0 ≤ reste < D n'est pas atteint.
uvec2 x_div(uvec4 N, uvec2 D, inout bool ok) {
    float df = x2float(D);
    uvec2 q = x2from(x4float(N)/df);
    for (int it = 0; it < 3; ++it) {
        float dq = x4float(x4sub(N, x2mul(q, D)))/df;
        uvec2 d = x2from(abs(dq) + 0.5);
        q = dq < 0.0 ? x2sub(q, d) : x2add(q, d);
    }
    for (int it = 0; it < 2; ++it) {
        uvec4 r = x4sub(N, x2mul(q, D));
        if (x4sign(r)) q = x2sub(q, uvec2(1u, 0u));
        else if (!x4lt(r, uvec4(D, 0u, 0u))) q = x2add(q, uvec2(1u, 0u));
    }
    uvec4 r = x4sub(N, x2mul(q, D));
    ok = ok && !x4sign(r) && x4lt(r, uvec4(D, 0u, 0u));
    return q;
}

// ln(v), v > 0 à l'échelle 2^-fraction, en S5.58.
uvec2 fx_ln(uvec4 v, int fraction) {
    int top = x4msb(v), e = top - fraction;
    uvec2 m = (top >= 63 ? x4shr(v, top - 63) : x4shl(v, 63 - top)).xy;          // Q1.63
    int j = int((m.y >> 25) & 63u);
    uvec2 d = x2sub(m, uvec2(0u, (uint(64 + j) << 25) | (1u << 24)));            // |d| ≤ 2^56
    uvec2 rd = x2hi(x2abs(d), fxt(FXT_INVC + j));
    uvec2 r = x2sign(d) ? x2neg(rd) : rd;                                         // S63
    uvec2 acc = fxt(FXT_LNK + 8);
    for (int k = 7; k >= 1; --k) acc = x2add(fxt(FXT_LNK + k), fx_smul(acc, r));
    uvec2 series = fx_smul(acc, r);                                               // S2.62
    uvec4 p = x2mul(fxt(FXT_CONST), uvec2(uint(abs(e)), 0u));                        // |e|·ln2 en Q.64
    uvec2 eln2 = x4sar(e < 0 ? x4neg(p) : p, 6).xy;
    return x2add(x2add(eln2, x2sar(fxt(FXT_LNC + j), 5)), x2sar(series, 4));
}

// e^y, y en S8.55 (y ≤ 5) : mantisse Q1.63, valeur = mant·2^(n-63).
uvec2 fx_exp(uvec2 y, out int n) {
    uvec4 p = x2mul(x2abs(y), fxt(FXT_CONST + 3));                                       // y·64/ln2 en Q.111
    int K = int(x4sar(x2sign(y) ? x4neg(p) : p, 111).x);
    uvec4 ln2 = uvec4(fxt(FXT_CONST + 5), fxt(FXT_CONST + 6));                                  // ln2/64 en Q0.90
    uvec4 r90 = x4sub(x4shl(x4ext(y), 35), x4muls(ln2, K));
    uvec2 r = x4sar(r90, 27).xy;                                                  // S63
    uvec2 acc = fxt(FXT_EXPK + 8);
    for (int k = 7; k >= 0; --k) acc = x2add(fxt(FXT_EXPK + k), fx_smul(acc, r));        // e^r en S2.62
    int q = K >= 0 ? K/64 : -((-K + 63)/64);
    int j = K - 64*q;
    uvec4 product = x2mul(fxt(FXT_POW2 + j), acc);                                       // Q3.125
    int shift = (product.w >> 30) != 0u ? 63 : 62;
    n = q + (shift - 62);
    return x4shr(product, shift).xy;
}

// EOTF : x > 0 à l'échelle 2^-fraction ; ok faux hors cadre (x ≳ 1,33) ou division non vérifiée.
// mant nul : x ≤ x0 au calcul. dl = ln(x/x0), kappa = conditionnement (num et den exacts).
uvec2 fx_eotf(uvec4 x, int fraction, out int n, out float dl, out float kappa, inout bool ok) {
    n = 0; dl = 0.0; kappa = 0.0;
    uvec2 dls = x2sub(fx_ln(x, fraction), fxt(FXT_CONST + 4));
    bool neg = x2sign(dls);
    dl = (neg ? -x2float(x2neg(dls)) : x2float(dls))*3.4694470e-18;              // 2^-58
    if (neg || (dls.x | dls.y) == 0u) return uvec2(0u);
    uvec2 delta = x4shr(x2mul(dls, fxt(FXT_CONST + 1)), 59).xy;                          // S63
    int j = int(delta.y >> 25);
    if (j > 15) { ok = false; return uvec2(0u); }
    uvec2 r0 = x2sub(delta, uvec2(0u, uint(j) << 25));
    uvec2 r = uvec2(r0.x << 1, (r0.y << 1) | (r0.x >> 31));                       // Q0.64
    uvec2 b = fxt(FXT_EM1K + 8);
    for (int k = 7; k >= 2; --k) b = x2add(fxt(FXT_EM1K + k), x2hi(b, r));
    uvec2 em = x2add(r, x2hi(x2hi(r, r), b));
    uvec2 ex = x2add(x2add(em, x2hi(fxt(FXT_EXPM1 + j), em)), fxt(FXT_EXPM1 + j));
    uvec2 num = x4shr(x2mul(ex, uvec2(107u, 0u)), 7).xy;                          // c1·expm1, Q0.64
    if (!x2lt(num, uvec2(0u, 43u << 24))) { ok = false; return uvec2(0u); }
    uvec2 den = x2sub(uvec2(0u, 6615u << 19), x4shr(x2mul(num, uvec2(299u, 0u)), 6).xy);   // Q2.62
    uvec2 g = x_div(x4shl(uvec4(num, 0u, 0u), 61), den, ok);                      // Q1.63
    float nf = x2float(num)*5.4210109e-20, df = x2float(den)*2.1684043e-19;
    kappa = (0.8359375 + nf)/12.56011962890625*(1.0/nf + 18.6875/df);
    if ((g.x | g.y) == 0u) return uvec2(0u);
    uvec2 lg = fx_ln(uvec4(g, 0u, 0u), 63);
    uvec4 py = x2mul(x2abs(lg), fxt(FXT_CONST + 2));
    uvec2 y = py.zw;
    if (x2sign(lg)) { if ((py.x | py.y) != 0u) y = x2add(y, uvec2(1u, 0u)); y = x2neg(y); }   // plancher
    return fx_exp(y, n);
}

// --- Premier niveau (lot L) : produits 32 × 32 bits, additions sur 64 bits (fixed_point.cpp, eotf_fast). ---
uint mulhi32(uint a, uint b) { uint h, l; umulExtended(a, b, h, l); return h; }
int imul_sar38(int a, int b) { int h, l; imulExtended(a, b, h, l); return h >> 6; }          // plancher(a·b / 2^38)
uint umul_shr38(uint a, uint b) { uint h, l; umulExtended(a, b, h, l); return h >> 6; }
// 64 bits ≪ k (0 ≤ k < 32) d'un mot de 32 bits.
uvec2 x2shl32(uint v, int k) { return k == 0 ? uvec2(v, 0u) : uvec2(v << k, v >> (32 - k)); }

// ln(v), v > 0 à l'échelle 2^-fraction, en S5.58 ; série de degré 4 en S1.30.
uvec2 fx_ln_fast(uvec4 v, int fraction) {
    int top = x4msb(v), e = top - fraction;
    uvec2 m = (top >= 63 ? x4shr(v, top - 63) : x4shl(v, 63 - top)).xy;          // Q1.63
    int j = int((m.y >> 25) & 63u);
    uvec2 d = x2sub(m, uvec2(0u, (uint(64 + j) << 25) | (1u << 24)));            // |d| ≤ 2^56
    int d32 = int((d.x >> 25) | (d.y << 7));                                      // plancher(d / 2^25)
    uint rm = mulhi32(d32 < 0 ? uint(-d32) : uint(d32), fxt(FXT_INVC + j).y);
    int r = d32 < 0 ? -int(rm) : int(rm);                                         // r·2^38
    int a = -(1 << 28);
    a = 357913941 + imul_sar38(a, r);
    a = -(1 << 29) + imul_sar38(a, r);
    a = (1 << 30) + imul_sar38(a, r);
    int h, l;
    imulExtended(a, r, h, l);
    uvec2 series = uvec2((uint(l) >> 10) | (uint(h) << 22), uint(h >> 10));       // S5.58
    uvec4 p = x2mul(fxt(FXT_CONST), uvec2(uint(abs(e)), 0u));
    uvec2 eln2 = x4sar(e < 0 ? x4neg(p) : p, 6).xy;
    return x2add(x2add(eln2, x2sar(fxt(FXT_LNC + j), 5)), series);
}

// e^y, y en S8.55 : mantisse Q1.63, valeur = mant·2^(n-63) ; série de degré 4 en Q1.31.
uvec2 fx_exp_fast(uvec2 y, out int n) {
    float yf = (x2sign(y) ? -x2float(x2neg(y)) : x2float(y))*2.7755576e-17;     // 2^-55
    int K = int(floor(yf*92.332482616893657));
    uvec4 ln2 = uvec4(fxt(FXT_CONST + 5), fxt(FXT_CONST + 6));                    // ln2/64 en Q0.90
    uvec4 r90 = x4sub(x4shl(x4ext(y), 35), x4muls(ln2, K));
    if (x4sign(r90)) { K -= 1; r90 = x4add(r90, ln2); }
    else if (!x4lt(r90, ln2)) { K += 1; r90 = x4sub(r90, ln2); }
    uint r38 = x4shr(r90, 52).x;                                                  // r·2^38
    uint a = 89478485u;
    a = 357913941u + umul_shr38(a, r38);
    a = 1073741824u + umul_shr38(a, r38);
    a = 2147483648u + umul_shr38(a, r38);
    uint h, l;
    umulExtended(a, r38, h, l);
    uint em = (h << 1) | (l >> 31);                                               // (e^r − 1)·2^38
    int q = K >= 0 ? K/64 : -((-K + 63)/64);
    uvec2 p2 = fxt(FXT_POW2 + (K - 64*q));
    uint h0, l0, h1, l1, c, c2, c3;
    umulExtended(p2.x, em, h0, l0);
    umulExtended(p2.y, em, h1, l1);
    uint mid = uaddCarry(h0, l1, c);
    uint top = h1 + c;                                                            // pow2·em = l0 + mid·2^32 + top·2^64
    uint lo = uaddCarry(p2.x, (mid >> 6) | (top << 26), c);
    uint hi = uaddCarry(p2.y, top >> 6, c2);
    hi = uaddCarry(hi, c, c3);
    n = q;
    if ((c2 | c3) != 0u) { n += 1; return uvec2((lo >> 1) | (hi << 31), (hi >> 1) | 0x80000000u); }
    return uvec2(lo, hi);
}

// EOTF rapide : comme fx_eotf, ln g = ln num − ln den ; err majore l'erreur relative de L.
// ok faux : hors cadre, ou voisinage du zéro de l'EOTF (le second niveau décide).
uvec2 fx_eotf_fast(uvec4 x, int fraction, out int n, out float dl, out float kappa, out float err, inout bool ok) {
    n = 0; dl = 0.0; kappa = 0.0; err = 0.0;
    uvec2 dls = x2sub(fx_ln_fast(x, fraction), fxt(FXT_CONST + 4));
    bool neg = x2sign(dls);
    dl = (neg ? -x2float(x2neg(dls)) : x2float(dls))*3.4694470e-18;              // 2^-58
    if (neg || (dls.x | dls.y) == 0u) return uvec2(0u);
    uvec2 delta = x4shr(x2mul(dls, fxt(FXT_CONST + 1)), 59).xy;                          // S63
    int j = int(delta.y >> 25);
    if (j > 15) { ok = false; return uvec2(0u); }
    uvec2 r0 = x2sub(delta, uvec2(0u, uint(j) << 25));
    uvec2 r = uvec2(r0.x << 1, (r0.y << 1) | (r0.x >> 31));                       // Q0.64, r < 1/64
    uvec2 em = uvec2(0u);
    if ((r.x | r.y) != 0u) {
        uint r38 = (r.x >> 26) | (r.y << 6);
        uint b = 35791394u;
        b = 178956971u + (mulhi32(b, r38) >> 6);
        b = 715827883u + (mulhi32(b, r38) >> 6);
        b = 2147483648u + (mulhi32(b, r38) >> 6);
        int sh = r.y != 0u ? 31 - findMSB(r.y) : 63 - findMSB(r.x);              // ≥ 6
        uint rn = sh < 32 ? (r.y << sh) | (r.x >> (32 - sh)) : r.x << (sh - 32);
        uint qv = mulhi32(mulhi32(rn, rn), b);                                    // r²·b = qv·2^-(32+2sh)
        int k = 32 - 2*sh;
        em = x2add(r, k >= 0 ? x2shl32(qv, k) : (-k < 32 ? uvec2(qv >> -k, 0u) : uvec2(0u)));
    }
    uvec2 ex = x2add(x2add(em, x2hi(fxt(FXT_EXPM1 + j), em)), fxt(FXT_EXPM1 + j));
    uvec2 num = x4shr(x2mul(ex, uvec2(107u, 0u)), 7).xy;                          // c1·expm1, Q0.64
    if (!x2lt(num, uvec2(0u, 43u << 24))) { ok = false; return uvec2(0u); }
    if ((num.x | num.y) == 0u) return uvec2(0u);
    uvec2 den = x2sub(uvec2(0u, 6615u << 19), x4shr(x2mul(num, uvec2(299u, 0u)), 6).xy);   // Q2.62
    float nf = x2float(num)*5.4210109e-20, df = x2float(den)*2.1684043e-19;
    kappa = (0.8359375 + nf)/12.56011962890625*(1.0/nf + 18.6875/df);
    uvec2 lg = x2sub(fx_ln_fast(uvec4(num, 0u, 0u), 64), fx_ln_fast(uvec4(den, 0u, 0u), 62));
    uvec4 py = x2mul(x2abs(lg), fxt(FXT_CONST + 2));
    uvec2 y = py.zw;
    if (x2sign(lg)) { if ((py.x | py.y) != 0u) y = x2add(y, uvec2(1u, 0u)); y = x2neg(y); }   // plancher
    // Borne (fixed_point.cpp) : ln 2^-35, expm1 2^-34, exp 2^-35 ; marge 1,25.
    float f = 1.0 + 18.6875*nf/df;
    float rnum = 2.9103830e-11*(1.0/dl + 0.01) + 5.8207661e-11 + 2.7755576e-17/dl + 8.6736174e-19;
    if (!(rnum <= 2.4414062e-4)) { ok = false; return uvec2(0u); }
    float B = 6.2774*(1.01*f*rnum + 5.8207661e-11) + 8.8817842e-16;
    err = 1.25*(B*(1.0 + B) + 1.01*2.9103830e-11);
    return fx_exp_fast(y, n);
}

uint fp(int i) { return fel_fp[i >> 2][i & 3]; }
uvec2 fp2(int i) { return uvec2(fp(i), fp(i + 1)); }
uvec4 fp4(int i) { return uvec4(fp(i), fp(i + 1), fp(i + 2), fp(i + 3)); }
float fpf(int i) { return uintBitsToFloat(fp(i)); }
