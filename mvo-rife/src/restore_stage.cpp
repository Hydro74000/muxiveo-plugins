// Restauration (R2) : étage en flux (voir restore_stage.h).
#include "restore_stage.h"

#include <algorithm>
#include <cmath>

using namespace restore_auto;

bool RestoreStage::init(RestoreNet* n, const FrameFormat& f, const ColorParams& c, const RestoreOptions& opt, std::string& error)
{
    net = n;
    fmt = f;
    color = c;
    o = opt;
    k = auto_k(o.k, o.animation);
    cuts_ = CutDetector(48, o.cut_look);
    const int shift = fmt.bit_depth - 8;
    if (color.full_range)
    {
        lo = 0;
        hi = (uint16_t)((1 << fmt.bit_depth) - 1);
    }
    else  // plage limitée : valeurs réservées exclues (10 bits : 4–1019, 8 bits : 1–254)
    {
        lo = (uint16_t)(1 << shift);
        hi = (uint16_t)((254 << shift) + (1 << shift) - 1);
    }
    // recadrage central ≤ 1920 × 1080 de la mesure du rapport (r1_ab_render.auto_ratio)
    crop.w = std::min(1920, fmt.width) / 2 * 2;
    crop.h = std::min(1080, fmt.height) / 2 * 2;
    crop.x0 = (fmt.width - crop.w) / 4 * 2;
    crop.y0 = (fmt.height - crop.h) / 4 * 2;
    if (fmt.bit_depth < 8 || fmt.bit_depth > 16)
    {
        error = "restauration : profondeur non prise en charge";
        return false;
    }
    return true;
}

void RestoreStage::samples(const std::vector<uint8_t>& frame, std::vector<uint16_t>& v) const
{
    const size_t n = (size_t)fmt.total_samples();
    v.resize(n);
    if (fmt.bytes_per_sample == 1)
        for (size_t i = 0; i < n; i++)
            v[i] = frame[i];
    else
        for (size_t i = 0; i < n; i++)
            v[i] = (uint16_t)(frame[2 * i] | (frame[2 * i + 1] << 8));
}

std::vector<uint16_t> RestoreStage::luma10(const std::vector<uint8_t>& frame) const
{
    // mesures (coupes, grain, rapport) sur l'échelle 10 bits du banc et du profil cible
    const size_t n = (size_t)fmt.luma_samples();
    std::vector<uint16_t> y(n);
    for (size_t i = 0; i < n; i++)
    {
        const unsigned v = fmt.bytes_per_sample == 1 ? frame[i] : (unsigned)(frame[2 * i] | (frame[2 * i + 1] << 8));
        y[i] = (uint16_t)(fmt.bit_depth <= 10 ? v << (10 - fmt.bit_depth) : v >> (fmt.bit_depth - 10));
    }
    return y;
}

bool RestoreStage::push(const uint8_t* frame)
{
    if (!err.empty())
        return false;
    std::vector<uint8_t> buf(fmt.padded_frame_bytes(), 0);
    std::copy(frame, frame + fmt.frame_bytes(), buf.begin());
    const std::vector<uint16_t> y = luma10(buf);
    cuts_.push(y.data(), fmt.width, fmt.height, fmt.width);
    src[n_in] = std::move(buf);
    n_in++;
    return pump();
}

bool RestoreStage::finish()
{
    if (!err.empty())
        return false;
    n_total = n_in;
    cuts_.finish();
    if (!pump())
        return false;
    if (next_out != n_in)
    {
        err = "restauration : trames restantes non rendues";
        return false;
    }
    return true;
}

bool RestoreStage::pop(std::vector<uint8_t>& frame)
{
    if (ready.empty())
        return false;
    frame = std::move(ready.front());
    ready.pop_front();
    return true;
}

std::optional<int> RestoreStage::shot_end(int s) const
{
    for (int t = s + 1; t <= cuts_.decided(); t++)
        if (cuts_.is_cut(t))
            return t;
    if (n_total && cuts_.decided() >= *n_total - 1)
        return *n_total;
    return std::nullopt;
}

bool RestoreStage::same_shot(int t, int u) const
{
    return u >= 0 && (!n_total || u < *n_total) && u <= cuts_.decided() && shot_start(u) == shot_start(t);
}

int RestoreStage::neighbour(int t, int d) const
{
    // voisine du même plan, en miroir aux bords et aux coupes (r1_ab_render.run.nb)
    if (same_shot(t, t + d) && d != 0)
        return t + d;
    return same_shot(t, t - d) ? t - d : t;
}

bool RestoreStage::shot_curve(int s)
{
    if (curves.count(s))
        return true;
    const std::optional<int> e = shot_end(s);
    const int stop = e ? std::min(*e, s + o.look) : s + o.look;
    if (cuts_.decided() < stop - 1 || n_in < stop)
        return false;
    const int ln = stop - s;
    const int step = std::max(1, std::min(2, ln / 3 ? ln / 3 : 1));
    std::vector<GrainCurve> cs;
    for (int t = s; t < stop; t += step)
    {
        const std::vector<uint16_t> y = luma10(src.at(t));
        cs.push_back(grain_curve(y.data(), fmt.width, fmt.height, fmt.width));
    }
    curves[s] = median_curve(cs);
    return true;
}

bool RestoreStage::run_net()
{
    const int u = next_net;
    const int lim = n_total ? std::min(u + 2, *n_total - 1) : u + 2;  // voisines t ± 2 lues et leurs coupes décidées
    if (u >= n_in || lim >= n_in || cuts_.decided() < lim)
        return false;
    const int s = shot_start(u);
    if (!shot_curve(s))
        return false;
    const std::vector<uint16_t> y = luma10(src.at(u));
    std::vector<uint8_t> map((size_t)fmt.width * fmt.height);
    grain_map(y.data(), fmt.width, fmt.height, fmt.width, curves.at(s), map.data());
    const int64_t nb[4] = {neighbour(u, -2), neighbour(u, -1), neighbour(u, 1), neighbour(u, 2)};
    for (int64_t i : {(int64_t)u, nb[0], nb[1], nb[2], nb[3]})
        if (!net->add_frame(i, src.at(i).data(), err))
            return false;
    const bool want = s <= u && u <= s + 10;  // images de mesure du rapport (plan court : à partir de s)
    std::vector<float> yuv, flows;
    if (!net->process(u, nb, map.data(), yuv, want ? &crop : nullptr, want ? &flows : nullptr, err))
        return false;
    // σ du résidu dans l'échelle de la trame (rapport d'écarts types : indépendant de la profondeur)
    std::vector<uint16_t> sv;
    samples(src.at(u), sv);
    sig[u] = residual_sigma(sv.data(), fmt.width, yuv.data(), fmt.width, fmt.width, fmt.height);
    out[u] = std::move(yuv);
    if (want)
    {
        std::vector<float> cl((size_t)crop.w * crop.h);
        for (int r = 0; r < crop.h; r++)
            for (int x = 0; x < crop.w; x++)
                cl[(size_t)r * crop.w + x] = y[(size_t)(crop.y0 + r) * fmt.width + crop.x0 + x];
        ana_luma[u] = std::move(cl);
        ana_flows[u] = std::move(flows);
    }
    next_net = u + 1;
    for (auto it = src.begin(); it != src.end() && it->first < next_net - 2; ++it)
        net->drop_frame(it->first);
    return true;
}

bool RestoreStage::shot_ratio(int s)
{
    if (ratios.count(s))
        return true;
    const std::optional<int> e = shot_end(s);
    if (!e && next_net <= s + 10)
        return false;
    std::vector<int> centers;
    const int ln = e ? *e - s : 1 << 30;
    if (ln >= 11)
        centers = {s + 4, s + 6, s + 8};
    else
    {
        for (int c : {s + 4, s + 6, s + 8})
            if (c + 2 <= *e - 1)
                centers.push_back(c);
        if (centers.empty() && ln >= 5)
            centers.push_back((s + 2 + *e - 3) / 2);
    }
    if (!centers.empty() && next_net <= *std::max_element(centers.begin(), centers.end()) + 2)
        return false;
    std::optional<double> r;
    if (!centers.empty())
    {
        std::map<int, AnalysisFrame> frames;
        for (const auto& [t, l] : ana_luma)
            frames[(int)t] = AnalysisFrame{l.data(), ana_flows.at(t).data()};
        r = ratio_from_sigma(sigma_bins(frames, centers, crop.w, crop.h, 1.0), fmt.width);
    }
    if (!r)
        r = last_ratio;  // plan trop court ou sans mesure : rapport du plan précédent
    ratios[s] = r;
    if (r)
        last_ratio = r;
    const int64_t until = e ? *e : s + 11;
    for (auto it = ana_luma.begin(); it != ana_luma.end() && it->first < until;)
    {
        ana_flows.erase(it->first);
        it = ana_luma.erase(it);
    }
    return true;
}

bool RestoreStage::emit()
{
    const int t = next_out;
    if (t >= next_net)
        return false;
    const int s = shot_start(t);
    const std::optional<int> e = shot_end(s);
    const int hi_t = e ? std::min(*e - 1, t + o.dose_look) : t + o.dose_look;
    if (next_net <= hi_t || !shot_ratio(s))
        return false;
    const std::optional<double> r = ratios.at(s);
    std::vector<float> window;
    for (int u = std::max(s, t - o.dose_look); u <= hi_t; u++)
        window.push_back(sig.at(u));
    const double eq = detail::percentile(window, 50.0) / sig.at(t);  // égalisation sur le plan
    const double part = auto_part(r, k);
    const double kk = part * eq;
    std::vector<uint16_t> sv, dv;
    samples(src.at(t), sv);
    dv.resize(sv.size());
    dose(sv.data(), out.at(t).data(), sv.size(), kk, lo, hi, dv.data(), o.dither_step > 0 ? t : -1, 0, std::max(1, o.dither_step));
    std::vector<uint8_t> frame(fmt.padded_frame_bytes(), 0);
    if (fmt.bytes_per_sample == 1)
        for (size_t i = 0; i < dv.size(); i++)
            frame[i] = (uint8_t)dv[i];
    else
        for (size_t i = 0; i < dv.size(); i++)
        {
            frame[2 * i] = (uint8_t)(dv[i] & 0xff);
            frame[2 * i + 1] = (uint8_t)(dv[i] >> 8);
        }
    ready.push_back(std::move(frame));
    logs[t] = Log{r, part, kk};
    next_out = t + 1;
    release_before(std::min(next_out, next_net - 2), next_out);
    return true;
}

void RestoreStage::release_before(int64_t keep_src, int64_t keep_out)
{
    for (auto it = src.begin(); it != src.end() && it->first < keep_src;)
        it = src.erase(it);
    for (auto it = out.begin(); it != out.end() && it->first < keep_out;)
        it = out.erase(it);
    for (auto it = sig.begin(); it != sig.end() && it->first < keep_out - o.dose_look;)
        it = sig.erase(it);
}

bool RestoreStage::pump()
{
    for (;;)
    {
        for (; cuts_scanned <= cuts_.decided() && cuts_scanned < cuts_.frames(); cuts_scanned++)  // débuts de plan décidés
            if (cuts_.is_cut(cuts_scanned))
                cut_list.push_back(cuts_scanned);
        const bool progressed = run_net() || emit();
        if (!err.empty())
            return false;
        held_max = std::max(held_max, src.size() + out.size());
        if (!progressed)
            return true;
    }
}
