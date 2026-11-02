// Restauration (R2) : étage en flux, placé avant l'interpolation. Portage de StreamAuto (r1r_auto_stream.py, banc,
// plan Restauration § 5 bis, jalon P2) : chaque trame est lue une fois, dans l'ordre ; chaque décision n'utilise qu'une
// fenêtre bornée, aucune passe préalable sur le film.
//
// Étapes, chacune dès que sa fenêtre est complète : écart de luma au quart → décision de coupe (t + cut_look) → courbe
// de grain du plan (images [s, s + look)) → réseau R2 de l'image u (coupes décidées jusqu'à u + 2 ; voisines du même
// plan, miroir aux bords et aux coupes) → σ du résidu → rapport de grain du plan (flux du réseau des images s … s + 10,
// formule bornée, biais selon la largeur) → dosage de t (égalisation sur [t − dose_look, t + dose_look] ∩ plan, un seul
// arrondi, tramage). Trames de sortie au format d'entrée, dans l'ordre.
//
// Recouvrement : le travail GPU d'une image (envoi, réseau, rapatriement, σ du résidu) tourne dans un fil dédié, un seul
// à la fois, pendant que l'étage prépare la carte de grain suivante et dose les images précédentes ; seul ce fil
// utilise le réseau. Sorties du réseau gardées en virgule fixe 16 bits (2^(16 − profondeur) pas par code).
#pragma once
#include <array>
#include <cstdint>
#include <deque>
#include <memory>
#include <future>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "restore_auto.h"
#include "restore_net.h"

struct RestoreOptions
{
    double k = 1.0;          // niveau : Fort 0,5, Auto / Moyen 1, Léger 1,5
    bool animation = false;  // contenu animé : un cran vers Fort
    int dither_step = 1;     // 0 : arrondi simple ; 1 : tramage ±1 code ; 4 : ±1 pas 8 bits (sortie 10 bits vers un encodeur 8 bits)
    int look = 24;           // courbe de grain : images [s, s + look) du plan
    int dose_look = 24;      // égalisation : [t − dose_look, t + dose_look] ∩ plan
    int cut_look = 12;       // décision de coupe : écarts jusqu'à t + cut_look
};

class RestoreStage
{
public:
    ~RestoreStage();
    // net : réseau configuré pour le format ; fmt : format des trames (8 à 16 bits, 4:2:0, 4:2:2 ou 4:4:4).
    bool init(RestoreNet* net, const FrameFormat& fmt, const ColorParams& color, const RestoreOptions& opt, std::string& error);
    // Trame brute (frame_bytes octets, tampon recopié). Faux en cas d'erreur (error()).
    bool push(const uint8_t* frame);
    // Fin du flux : toutes les trames restantes deviennent disponibles.
    bool finish();
    // Prochaine trame restaurée (format d'entrée, tampon padded_frame_bytes), si disponible.
    bool pop(std::vector<uint8_t>& frame);
    const std::string& error() const { return err; }
    // Diagnostic : trames tenues (sources + sorties du réseau) au plus ; tuiles après repli éventuel.
    size_t max_held() const { return held_max; }
    int tile_size() const { return tile; }
    // Journal : rapport et part réinjectée par image (premier niveau).
    struct Log
    {
        std::optional<double> ratio;
        double part = 0.0, k_eff = 0.0;
    };
    const std::map<int64_t, Log>& log() const { return logs; }
    const std::vector<int>& cuts() const { return cut_list; }

private:
    struct Job
    {
        bool ok = false;
        std::string error;
        std::vector<uint16_t> out;  // virgule fixe
        std::vector<float> flows;
        int tile = 0;
    };
    bool pump(bool block);
    bool prepare();
    bool launch();
    bool collect();
    bool emit();
    bool shot_curve(int s);
    bool shot_ratio(int s);
    int shot_start(int t) const { return cuts_.shot_start(t); }
    std::optional<int> shot_end(int s) const;
    bool same_shot(int t, int u) const;
    int neighbour(int t, int d) const;
    std::vector<uint16_t> luma10(const std::vector<uint8_t>& frame) const;
    void samples(const uint8_t* frame, std::vector<uint16_t>& out) const;
    void release();

    RestoreNet* net = nullptr;
    FrameFormat fmt;
    ColorParams color;
    RestoreOptions o;
    double k = 1.0;
    uint16_t lo = 0, hi = 0;
    float fx_scale = 64.f;  // pas de la virgule fixe par code
    int tile = 512;
    size_t max_hold = 80;  // contre-pression : trames sources tenues au plus avant d'attendre le GPU
    RestoreNet::Crop crop;
    restore_auto::CutDetector cuts_;
    std::vector<int> cut_list;
    int cuts_scanned = 0;
    std::optional<int> n_total;
    int n_in = 0, next_net = 0, done_net = 0, next_out = 0;
    std::future<Job> job;
    int job_u = -1;
    int pending_sig = -1;  // image collectée dont le σ du résidu reste à calculer
    int prepared_u = -1;   // image dont la carte de grain est prête (prepared_map)
    std::shared_ptr<std::vector<uint8_t>> prepared_map;
    bool job_want = false;
    int64_t net_low = 0;  // trames déjà retirées du cache du réseau (fil GPU seulement)
    std::map<int64_t, std::vector<uint8_t>> src;
    std::map<int64_t, std::vector<uint16_t>> out;
    std::map<int64_t, float> sig;
    std::map<int, restore_auto::GrainCurve> curves;
    std::map<int64_t, std::vector<float>> ana_luma, ana_flows;
    std::map<int, std::optional<double>> ratios;
    std::optional<double> last_ratio;
    std::deque<std::vector<uint8_t>> ready;
    std::map<int64_t, Log> logs;
    size_t held_max = 0;
    std::string err;
};
