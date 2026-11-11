// Restauration (R2) : étage en flux (voir restore_stage.h).
#include "restore_stage.h"

#include "gpu_watchdog.h"

#include <algorithm>
#include <cmath>

using namespace restore_auto;

RestoreStage::~RestoreStage()
{
    if (worker.joinable())
    {
        {
            std::lock_guard<std::mutex> lk(wm);
            worker_stop = true;
        }
        wcv.notify_all();
        worker.join();
    }
}

void RestoreStage::gpu_worker()
{
    std::unique_lock<std::mutex> lk(wm);
    for (;;)
    {
        wcv.wait(lk, [&] { return worker_stop || !todo.empty(); });
        if (worker_stop)
            return;
        Item it = std::move(todo.front());
        todo.pop_front();
        lk.unlock();
        Job r;
        run_item(it, r);
        lk.lock();
        results.push_back(std::move(r));
        wcv.notify_all();
    }
}

void RestoreStage::run_item(const Item& it, Job& r)
{
    r.u = it.u;
    r.want = it.want;
    for (const auto& [index, data] : it.inputs)
        if (!net->add_frame(index, data, r.error))
            return;
    // mémoire GPU insuffisante : repli sur des tuiles plus petites au lieu d'un échec
    for (;;)
    {
        if (net->process(it.u, it.nb.data(), it.map->data(), r.yuv, it.want ? &crop : nullptr, it.want ? &r.flows : nullptr, r.error))
            break;
        if (!net->out_of_memory() || net->tile_size() <= 128)
            return;
        net->set_tile(net->tile_size() / 2);
    }
    r.tile = net->tile_size();
    for (; net_low < it.u - 2; net_low++)  // trames dont le réseau n'a plus besoin
        net->drop_frame(net_low);
    r.ok = true;
    watchdog::progress();
}

bool RestoreStage::init(RestoreNet* n, const FrameFormat& f, const ColorParams& c, const RestoreOptions& opt, std::string& error)
{
    net = n;
    fmt = f;
    color = c;
    o = opt;
    k = auto_k(o.k, o.animation);
    cuts_ = CutDetector(48, o.cut_look);
    if (fmt.bit_depth < 8 || fmt.bit_depth > 16)
    {
        error = "restauration : profondeur non prise en charge";
        return false;
    }
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
    fx_scale = (float)(1 << (16 - fmt.bit_depth));
    tile = net->tile_size();
    max_hold = (size_t)(o.look + o.cut_look + o.dose_look + 20);
    // recadrage central ≤ 1920 × 1080 de la mesure du rapport (r1_ab_render.auto_ratio)
    crop.w = std::min(1920, fmt.width) / 2 * 2;
    crop.h = std::min(1080, fmt.height) / 2 * 2;
    crop.x0 = (fmt.width - crop.w) / 4 * 2;
    crop.y0 = (fmt.height - crop.h) / 4 * 2;
    return true;
}

std::vector<uint8_t> RestoreStage::take_buffer(std::vector<std::vector<uint8_t>>& spare) const
{
    if (spare.empty())
        return std::vector<uint8_t>(fmt.padded_frame_bytes(), 0);
    std::vector<uint8_t> b = std::move(spare.back());
    spare.pop_back();
    return b;
}

std::vector<uint16_t> RestoreStage::luma10(const std::vector<uint8_t>& frame) const
{
    // mesures (coupes, grain, rapport) sur l'échelle 10 bits du banc et du profil cible
    const size_t n = (size_t)fmt.luma_samples();
    std::vector<uint16_t> y(n);
    detail::for_rows(fmt.height, [&](int r0, int r1) {
        for (size_t i = (size_t)r0 * fmt.width; i < (size_t)r1 * fmt.width; i++)
        {
            const unsigned v = fmt.bytes_per_sample == 1 ? frame[i] : (unsigned)(frame[2 * i] | (frame[2 * i + 1] << 8));
            y[i] = (uint16_t)(fmt.bit_depth <= 10 ? v << (10 - fmt.bit_depth) : v >> (fmt.bit_depth - 10));
        }
    });
    return y;
}

bool RestoreStage::push(const uint8_t* frame)
{
    if (!err.empty())
        return false;
    // tampon recyclé (marge de fin déjà à zéro, jamais écrite) ou neuf
    std::vector<uint8_t> buf = take_buffer(spare_src);
    std::copy(frame, frame + fmt.frame_bytes(), buf.begin());
    const std::vector<uint16_t> y = luma10(buf);
    cuts_.push(y.data(), fmt.width, fmt.height, fmt.width);
    src[n_in] = std::move(buf);
    n_in++;
    // contre-pression : au-delà de max_hold trames tenues, attendre le GPU
    return pump(src.size() > max_hold);
}

bool RestoreStage::finish()
{
    if (!err.empty())
        return false;
    n_total = n_in;
    cuts_.finish();
    if (!pump(true))
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
    // échange : le tampon rendu par l'appelant (trame précédente, déjà recopiée) sert à une prochaine sortie
    frame.swap(ready.front());
    if (ready.front().size() == fmt.padded_frame_bytes())
        spare_out.push_back(std::move(ready.front()));
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

bool RestoreStage::prepare()
{
    // carte de grain et luma de mesure de la prochaine image à lancer, calculées d'avance (pendant le travail GPU)
    const int u = next_net;
    if (prepared_u == u)
        return false;
    const int lim = n_total ? std::min(u + 2, *n_total - 1) : u + 2;  // voisines t ± 2 lues et leurs coupes décidées
    if (u >= n_in || lim >= n_in || cuts_.decided() < lim)
        return false;
    const int s = shot_start(u);
    if (!shot_curve(s))
        return false;
    const std::vector<uint16_t> y = luma10(src.at(u));
    prepared_map = std::make_shared<std::vector<uint8_t>>((size_t)fmt.width * fmt.height);
    grain_map(y.data(), fmt.width, fmt.height, fmt.width, curves.at(s), prepared_map->data());
    if (s <= u && u <= s + 10)  // images de mesure du rapport (plan court : à partir de s)
    {
        std::vector<float> cl((size_t)crop.w * crop.h);
        for (int r = 0; r < crop.h; r++)
            for (int x = 0; x < crop.w; x++)
                cl[(size_t)r * crop.w + x] = y[(size_t)(crop.y0 + r) * fmt.width + crop.x0 + x];
        ana_luma[u] = std::move(cl);
    }
    prepared_u = u;
    return true;
}

bool RestoreStage::launch()
{
    if (in_flight >= MAX_IN_FLIGHT)
        return false;
    const int u = next_net;
    if (prepared_u != u && !prepare())
        return false;
    const int s = shot_start(u);
    Item it;
    it.u = u;
    it.map = std::move(prepared_map);
    prepared_u = -1;
    it.nb = {neighbour(u, -2), neighbour(u, -1), neighbour(u, 1), neighbour(u, 2)};
    it.want = s <= u && u <= s + 10;
    // trames tenues par le travail : pointeurs stables (aucune n'est retirée avant la collecte, voir release())
    for (int64_t i : {(int64_t)u, it.nb[0], it.nb[1], it.nb[2], it.nb[3]})
        it.inputs.emplace_back(i, src.at(i).data());
    next_net = u + 1;
    in_flight++;
    if (!worker.joinable())
        worker = std::thread([this] { gpu_worker(); });
    {
        std::lock_guard<std::mutex> lk(wm);
        todo.push_back(std::move(it));
    }
    wcv.notify_all();
    return true;
}

int RestoreStage::collect(bool block)
{
    if (pending_sig >= 0 || in_flight == 0)  // σ de l'image précédente d'abord
        return 0;
    Job r;
    {
        std::unique_lock<std::mutex> lk(wm);
        if (results.empty() && !block)
            return 0;
        wcv.wait(lk, [&] { return !results.empty(); });
        r = std::move(results.front());
        results.pop_front();
    }
    in_flight--;
    next_collect = r.u + 1;
    if (!r.ok)
    {
        err = r.error.empty() ? "restauration : échec du réseau" : r.error;
        return -1;
    }
    tile = r.tile;
    // sortie déjà en virgule fixe (convertie sur le GPU) ; σ du résidu ensuite, dans ce fil, pendant que le fil GPU
    // calcule les images suivantes
    out[r.u] = r.yuv;
    pending_sig = r.u;
    if (r.want)
        ana_flows[r.u] = std::move(r.flows);
    return 1;
}

bool RestoreStage::shot_ratio(int s)
{
    if (ratios.count(s))
        return true;
    const std::optional<int> e = shot_end(s);
    if (!e && done_net <= s + 10)
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
    if (!centers.empty() && done_net <= *std::max_element(centers.begin(), centers.end()) + 2)
        return false;
    std::optional<double> r;
    if (!centers.empty())
    {
        std::map<int, AnalysisFrame> frames;
        for (const auto& [t, l] : ana_luma)
            if (ana_flows.count(t))
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
    if (t >= done_net)
        return false;
    const int s = shot_start(t);
    const std::optional<int> e = shot_end(s);
    const int hi_t = e ? std::min(*e - 1, t + o.dose_look) : t + o.dose_look;
    if (done_net <= hi_t || !shot_ratio(s))
        return false;
    const std::optional<double> r = ratios.at(s);
    std::vector<float> window;
    for (int u = std::max(s, t - o.dose_look); u <= hi_t; u++)
        window.push_back(sig.at(u));
    const double eq = detail::percentile(window, 50.0) / sig.at(t);  // égalisation sur le plan
    const double part = auto_part(r, k);
    // part réinjectée plafonnée à 1 : la sortie reste entre l'image débruitée et la source. Sans plafond, une image dont le
    // résidu est bien plus faible que celui du plan (σ 0,08 face à 0,48 : fin d'un plan, image presque propre) recevait
    // 6,8 fois l'écart source − sortie : points saturés (10 000 nits en PQ), MaxCLL mesuré faussé (2026-11-11)
    const double kk = std::min(1.0, part * eq);
    std::vector<uint8_t> frame = take_buffer(spare_out);
    dose_frame(src.at(t).data(), fmt.bytes_per_sample, (const uint16_t*)out.at(t).data, 1.f / fx_scale, (size_t)fmt.total_samples(), kk, lo, hi,
               frame.data(), o.dither_step > 0 ? t : -1, std::max(1, o.dither_step));
    ready.push_back(std::move(frame));
    logs[t] = Log{r, part, kk};
    next_out = t + 1;
    release();
    return true;
}

void RestoreStage::release()
{
    // sources : gardées pour le dosage (next_out) et pour les voisines du réseau (dernier lancé − 2)
    const int64_t keep_src = std::min<int64_t>(next_out, (in_flight > 0 ? next_collect : next_net) - 2);
    for (auto it = src.begin(); it != src.end() && it->first < keep_src;)
    {
        if (spare_src.size() < 4)
            spare_src.push_back(std::move(it->second));
        it = src.erase(it);
    }
    for (auto it = out.begin(); it != out.end() && it->first < next_out;)
        it = out.erase(it);
    for (auto it = sig.begin(); it != sig.end() && it->first < next_out - o.dose_look;)
        it = sig.erase(it);
}

bool RestoreStage::pump(bool block)
{
    for (;;)
    {
        for (; cuts_scanned <= cuts_.decided() && cuts_scanned < cuts_.frames(); cuts_scanned++)  // débuts de plan décidés
            if (cuts_.is_cut(cuts_scanned))
                cut_list.push_back(cuts_scanned);
        bool progressed = false;
        const int c = collect(false);
        if (c < 0)
            return false;
        if (c > 0)
            progressed = true;
        if (launch())
            progressed = true;
        if (in_flight > 0 && prepare())  // image suivante préparée pendant le travail GPU
            progressed = true;
        if (pending_sig >= 0)
        {
            sig[pending_sig] = residual_sigma_fx(src.at(pending_sig).data(), fmt.bytes_per_sample, (const uint16_t*)out.at(pending_sig).data,
                                                 1.f / fx_scale, fmt.width, fmt.height);
            done_net = pending_sig + 1;
            pending_sig = -1;
            progressed = true;
        }
        while (emit())
            progressed = true;
        held_max = std::max(held_max, src.size() + out.size());
        if (!err.empty())
            return false;
        if (progressed)
            continue;
        // rien d'autre à faire : attendre le GPU si l'appelant l'exige (fin du flux, contre-pression)
        if (in_flight > 0 && (block || src.size() > max_hold))
        {
            if (collect(true) < 0)
                return false;
            continue;
        }
        return true;
    }
}
