// SPDX-License-Identifier: LGPL-2.1-or-later
// Rendu exact d'un pixel en virgule fixe (lot A2') : traduction de fixed::pixel (src/fixed_point.cpp)
// à partir des entrées entières exactes (fel_inputs). Codes candidats dans out3, masque des
// composantes non certaines en retour ; R (Q0.96), bornes et codes pour le diagnostic.
// Décalages des paramètres : fixed_abi.h (FX_CURVE0, FX_CURVE, FX_GLOBAL définis avant ce fichier).

// Seuil T_bas·2^96 (trois mots) ; fx_threshold_up : suivant(T_bas), au-dessus du seuil exact (le
// quatrième mot de la texture porte l'exposant de l'ulp). R n'étant pas un double, la borne basse se
// compare à suivant(T_bas).
uvec4 fx_threshold(int k) { uvec4 t = texelFetch(fel_thr96, ivec2(k & 255, k >> 8), 0); return uvec4(t.xyz, 0u); }
uvec4 fx_threshold_up(int k) {
    uvec4 t = texelFetch(fel_thr96, ivec2(k & 255, k >> 8), 0);
    return x4add(uvec4(t.xyz, 0u), x4shl(uvec4(1u, 0u, 0u, 0u), int(t.w)));
}

uint fx_render(int code, int tb, int tr, int Db, int te0, int te1, int te2, int Dl, int Dc,
               out int seg[3], out vec3 out3, out uvec4 Rd[3], out float Ed[3], out uint qd[3]) {
    const float U96 = 1.2621774e-29;                                              // 2^-96
    int d = int(fp(0)), q = int(fp(1));
    bool residual = fp(2) != 0u;
    int scale = 64 + d, fraction = scale + q;
    bool ok = true;
    uint h, l;
    umulExtended(1023u, uint(Db), h, l);
    uvec2 full = uvec2(l, h);                                                     // 1023·Db
    int sums[3] = int[3](code, max(tb, 0), max(tr, 0));
    // Segments : comparaisons entières exactes.
    for (int c = 0; c < 3; ++c) {
        int B = FX_CURVE0 + c*FX_CURVE, count = int(fp(B)), p = 0;
        while (p + 2 < count) {
            uint pc = fp(B + 2 + p);
            bool ge;
            if (c == 0) ge = uint(code) >= pc;
            else {
                uvec2 n = uvec2(uint(sums[c]), 0u);
                if (!x2lt(n, full)) n = full;
                umulExtended(pc, uint(Db), h, l);
                ge = !x2lt(n, uvec2(l, h));
            }
            if (!ge) break;
            ++p;
        }
        seg[c] = p;
    }
    // Signaux Q0.64 (plancher ; 1 → 2^64 - 1) et monômes tronqués.
    // Puissances des monômes calculées à la volée (pas de tableau indexé par l'ordre MMR).
    uvec2 t[7];
    t[0] = code >= 1023 ? uvec2(0xffffffffu) : x_div(uvec4(0u, 0u, uint(code), 0u), uvec2(1023u, 0u), ok);
    for (int c = 1; c < 3; ++c) {
        uvec2 n = uvec2(uint(sums[c]), 0u);
        t[c] = x2lt(n, full) ? x_div(uvec4(0u, 0u, n), full, ok) : uvec2(0xffffffffu);
    }
    t[3] = x2hi(t[0], t[1]); t[4] = x2hi(t[0], t[2]); t[5] = x2hi(t[1], t[2]); t[6] = x2hi(t[3], t[2]);
    // Mapping, bornage, NLQ et décalage à l'échelle 2^-(64+d).
    uvec4 v[3];
    float ev[3];
    int te[3] = int[3](te0, te1, te2);
    for (int c = 0; c < 3; ++c) {
        int B = FX_CURVE0 + c*FX_CURVE, k = seg[c], base = B + 26 + 24*k;
        int c0 = int(fp(base));
        uvec4 s = uvec4(0u, 0u, uint(c0), c0 < 0 ? 0xffffffffu : 0u);
        if (fp(B + 10 + k) == 0u)
            s = x4add(x4add(s, x2muls(t[c], int(fp(base + 1)))), x2muls(x2hi(t[c], t[c]), int(fp(base + 2))));
        else {
            int order = int(fp(B + 18 + k));
            for (int j = 0; j < 7; ++j) {
                uvec2 p1 = t[j];
                s = x4add(s, x2muls(p1, int(fp(base + 1 + j))));
                if (order > 1) {
                    uvec2 p2 = x2hi(p1, p1);
                    s = x4add(s, x2muls(p2, int(fp(base + 8 + j))));
                    if (order > 2) s = x4add(s, x2muls(x2hi(p2, p1), int(fp(base + 15 + j))));
                }
            }
        }
        int count = int(fp(B));
        uvec4 s1023 = x4muls(s, 1023);
        if (x4lt(s1023, x4shl(uvec4(fp(B + 1), 0u, 0u, 0u), scale))) s = fp4(B + 218);
        else if (x4lt(x4shl(uvec4(fp(B + count), 0u, 0u, 0u), scale), s1023)) s = fp4(B + 222);
        ev[c] = fpf(B + 226 + k);
        if (residual && te[c] != 0) {
            uvec2 De = uvec2(uint(c == 0 ? Dl : Dc), 0u);
            umulExtended(uint(abs(te[c])), fp(FX_GLOBAL + 2*c), h, l);
            uvec2 prod = uvec2(l, h);
            uvec2 a = x_div(uvec4(prod, 0u, 0u), De, ok);
            uvec2 rem = x2sub(prod, x2mul(a, De).xy);
            uvec2 frac = x_div(uvec4(0u, 0u, rem), De, ok);
            uvec4 qn = x4add(uvec4(frac, a), x4shl(x4ext(fp2(FX_GLOBAL + 6 + 2*c)), 63));
            s = te[c] > 0 ? x4add(s, qn) : x4sub(s, qn);
            ev[c] += exp2(-float(scale));
        }
        v[c] = x4sub(s, fp4(FX_GLOBAL + 12 + 4*c));
    }
    // Signaux LMS' (échelle 2^-fraction), EOTF et encadrement relatif.
    bool unc = !ok;
    uvec4 Lq[3];
    float Lf[3], rel[3];
    float sq = exp2(-float(q)), sf = exp2(-float(fraction));
    for (int i = 0; i < 3; ++i) {
        uvec4 x = x4add(x4add(x4muls(v[0], int(fp(FX_GLOBAL + 24 + 3*i))), x4muls(v[1], int(fp(FX_GLOBAL + 25 + 3*i)))),
                        x4muls(v[2], int(fp(FX_GLOBAL + 26 + 3*i))));
        float ex = sf;
        for (int j = 0; j < 3; ++j) ex += abs(float(int(fp(FX_GLOBAL + 24 + 3*i + j))))*sq*ev[j];
        ex *= 1.01;
        float xv = x4float(x)*sf;
        Lq[i] = uvec4(0u); Lf[i] = 0.0; rel[i] = 0.0;
        bool zero = xv*(1.0 + 1e-5) + ex <= 7.30955e-7;
        if (x4sign(x) || x4zero(x) || zero) { if (!zero) unc = true; continue; }
        int n; float dl, kappa; bool okx = true;
        uvec2 mant = fx_eotf(x, fraction, n, dl, kappa, okx);
        if (!okx) { unc = true; continue; }
        if ((mant.x | mant.y) == 0u) { if (!zero) unc = true; continue; }
        if (!(xv*(1.0 - 1e-5) - ex > 7.30957e-7 && kappa*ex <= 1e-6*xv && ex <= 1e-6*xv && dl > 0.0)) { unc = true; continue; }
        rel[i] = 8.8817842e-16*(1.0 + 1.0/dl) + 1.01*kappa*ex/xv;                // 2^-50
        Lf[i] = x2float(mant)*exp2(float(n - 63));
        int sh = n + 33;
        Lq[i] = sh >= 0 ? x4shl(uvec4(mant, 0u, 0u), sh) : (sh > -64 ? x4shr(uvec4(mant, 0u, 0u), -sh) : uvec4(0u));
    }
    // Matrice LMS→RGB (S2.61) en Q0.96, bornes, classement exact sur les seuils Q0.96.
    uint mask = unc ? 7u : 0u;
    for (int c = 0; c < 3; ++c) {
        uvec4 R = uvec4(0u);
        float e = 5.0*U96;
        for (int k = 0; k < 3; ++k) {
            uvec2 m = fp2(FX_GLOBAL + 33 + 2*(3*c + k)), am = x2abs(m);
            uvec4 prod = x4add(x4shl(x2mul(am, Lq[k].zw), 3), x4shr(x2mul(am, Lq[k].xy), 61));
            R = x2sign(m) ? x4sub(R, prod) : x4add(R, prod);
            float merr = fpf(FX_GLOBAL + 51 + 3*c + k);
            e += ((x2float(am)*4.3368087e-19 + merr)*rel[k] + merr)*Lf[k]*1.000001 + 2.0*U96;
        }
        e *= 1.001;
        float Rf = x4float(R)*U96;
        int qc = Rf <= 0.0 ? 0 : Rf >= 1.0 ? 65535 : clamp(int(floor(g_oetf(Rf)*65535.0 + 0.5)), 0, 65535);
        for (int i = 0; i < 8 && qc < 65535 && x4lt(fx_threshold(qc), R); ++i) ++qc;
        for (int i = 0; i < 8 && qc > 0 && !x4lt(fx_threshold(qc - 1), R); ++i) --qc;
        float e96 = e*7.9228163e28;
        uvec4 eq = uvec4(x2add(x2from(e96), uvec2(1u, 0u)), 0u, 0u);
        bool certain = !unc && e96 < 1.8e19 && (qc == 0 || !x4lt(x4sub(R, eq), fx_threshold_up(qc - 1))) &&
                       (qc == 65535 || !x4lt(fx_threshold(qc), x4add(R, eq)));
        if (!certain) mask |= 1u << c;
        out3[c] = (float(qc) + 0.25)/65535.0;
        Rd[c] = R; Ed[c] = e; qd[c] = uint(qc) | (certain ? 65536u : 0u);
    }
    return mask;
}
