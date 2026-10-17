/* SPDX-License-Identifier: LGPL-2.1-or-later */
/* Statistiques de l'image finale, référence CPU du shader GPU (voir include/frame_stats.h).
 * Unité compilée sans contraction (CMakeLists) : Y identique au GPU (« precise »). */
#include "frame_stats.h"

#include <math.h>
#include <string.h>

static float eotf_table[65536], edges_table[FEL_STATS_BINS + 1];
static int ready;

/* EOTF PQ (SMPTE ST 2084) en double, arrondie en float. */
static double pq(double e)
{
    const double m1 = 2610.0/16384, m2 = 2523.0/4096*128, c1 = 3424.0/4096, c2 = 2413.0/4096*32,
                 c3 = 2392.0/4096*32;
    const double p = pow(e, 1/m2), n = p - c1 > 0 ? p - c1 : 0;
    return 10000*pow(n/(c2 - c3*p), 1/m1);
}

static void prepare(void)
{
    if (ready) return;
    for (int k = 0; k < 65536; ++k) eotf_table[k] = (float)pq(k/65535.0);
    for (int i = 0; i <= FEL_STATS_BINS; ++i) edges_table[i] = i < FEL_STATS_BINS ? eotf_table[64*i] : 10000.0f;
    ready = 1;
}

const float *fel_stats_eotf(void) { prepare(); return eotf_table; }
const float *fel_stats_edges(void) { prepare(); return edges_table; }

void fel_stats_row(const uint16_t *r, const uint16_t *g, const uint16_t *b, int count,
                   uint32_t *hist_maxrgb, uint32_t *hist_y, uint32_t *row)
{
    const float *eotf = fel_stats_eotf(), *edges = fel_stats_edges();
    uint32_t mr = 0, mg = 0, mb = 0, le = 0;
    float sum = 0;
    for (int x = 0; x < count; ++x) {
        const uint32_t cr = r[x], cg = g[x], cb = b[x];
        const uint32_t m = cr > cg ? (cr > cb ? cr : cb) : (cg > cb ? cg : cb);
        if (cr > mr) mr = cr;
        if (cg > mg) mg = cg;
        if (cb > mb) mb = cb;
        ++hist_maxrgb[m >> 6];
        sum += eotf[m];
        const float y = 0.2627f*eotf[cr] + 0.678f*eotf[cg] + 0.0593f*eotf[cb];
        le += y <= 100.0f;
        /* Plus grande borne ≤ Y (recherche dichotomique, identique au shader). */
        uint32_t lo = 0, hi = FEL_STATS_BINS;
        while (hi - lo > 1) {
            const uint32_t mid = (lo + hi) >> 1;
            if (edges[mid] <= y) lo = mid; else hi = mid;
        }
        ++hist_y[lo];
    }
    uint32_t bits;
    memcpy(&bits, &sum, sizeof(bits));
    row[0] = mr; row[1] = mg; row[2] = mb; row[3] = bits; row[4] = le;
}

void fel_stats_reset(struct fel_frame_stats *stats)
{
    memset(stats, 0, sizeof(*stats));
}

void fel_stats_add_rows(struct fel_frame_stats *stats, const uint32_t *rows, int count, int width)
{
    for (int i = 0; i < count; ++i) {
        const uint32_t *row = rows + (size_t)FEL_STATS_ROW_WORDS*i;
        for (int c = 0; c < 3; ++c) if (row[c] > stats->max_code[c]) stats->max_code[c] = row[c];
        float sum;
        memcpy(&sum, &row[3], sizeof(sum));
        stats->sum_maxrgb += sum;
        stats->y_le_100 += row[4];
        stats->pixels += (uint64_t)width;
    }
}
