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
// Recouvrement : le travail GPU des images (envoi, réseau, rapatriement) tourne dans un fil GPU persistant, alimenté par
// une file de deux images d'avance ; seul ce fil utilise le réseau. Pendant ce temps, l'étage prépare les cartes de grain,
// convertit les résultats, calcule le σ du résidu et dose (calculs processeur sur une réserve de fils commune). Sorties
// du réseau gardées en virgule fixe 16 bits (2^(16 − profondeur) pas par code).
#pragma once
#include <array>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
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
    double sharpen = 0.0;    // contre-accentuage temporel après le dosage (0 : aucun ; voir restore_auto::SharpenParams)
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
    // Vrai si la trame t (déjà sortie de l'étage) commence un plan.
    bool shot_starts_at(int t) const { return shot_start(t) == t; }

private:
    // Image confiée au fil GPU : voisines, trames tenues par l'étage jusqu'à la collecte du résultat, carte de grain.
    struct Item
    {
        int u = 0;
        std::array<int64_t, 4> nb{};
        std::vector<std::pair<int64_t, const uint8_t*>> inputs;
        std::shared_ptr<std::vector<uint8_t>> map;
        bool want = false;
        unsigned flow_mask = 0;  // flux rapatriés pour la mesure (bit j : voisine j, ordre t − 2, t − 1, t + 1, t + 2)
    };
    // Résultat brut du fil GPU.
    struct Job
    {
        int u = -1;
        bool want = false;
        bool ok = false;
        std::string error;
        ncnn::Mat yuv;  // codes Y'CbCr en virgule fixe 16 bits, rapatriés
        ncnn::Mat ref;  // référence du contre-accentuage (luma, virgule fixe), si demandée
        ncnn::Mat flows[4];  // flux recadrés tels que rapatriés (plans de flow_mask)
        unsigned flow_mask = 0;
        int tile = 0;
    };
    bool pump(bool block);
    bool prepare();
    bool curve_ahead();
    std::vector<int> ratio_centers(int s, std::optional<int> e) const;
    unsigned measure_mask(int s, int u) const;
    bool launch();
    int collect(bool block);  // 1 : résultat collecté ; 0 : aucun ; −1 : erreur
    void gpu_worker();
    void run_item(const Item& it, Job& r);
    bool emit();
    bool shot_curve(int s);
    bool shot_ratio(int s);
    int shot_start(int t) const { return cuts_.shot_start(t); }
    std::optional<int> shot_end(int s) const;
    bool same_shot(int t, int u) const;
    int neighbour(int t, int d) const;
    std::vector<uint16_t> luma10(const std::vector<uint8_t>& frame) const;
    // Tampon de trame (padded_frame_bytes) : recyclé, marge de fin à zéro, ou neuf.
    std::vector<uint8_t> take_buffer(std::vector<std::vector<uint8_t>>& spare) const;
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
    size_t curve_scan = 0;  // prochain début de plan de cut_list dont la courbe peut être calculée d'avance
    std::optional<int> n_total;
    int n_in = 0, next_net = 0, done_net = 0, next_out = 0;
    // images confiées au fil GPU au plus : réserve de travail pendant les calculs processeur ponctuels de ce fil (rapport
    // de grain d'un plan, ≈ 0,5 s en 4K), sans quoi le GPU attend
    static constexpr int MAX_IN_FLIGHT = 8;
    std::thread worker;
    std::mutex wm;
    std::condition_variable wcv;
    std::deque<Item> todo;
    std::deque<Job> results;
    bool worker_stop = false;
    int in_flight = 0;     // images confiées au fil GPU dont le résultat n'est pas collecté
    int next_collect = 0;  // prochaine image attendue du fil GPU
    int pending_sig = -1;  // image collectée dont le σ du résidu reste à calculer
    int prepared_u = -1;   // image dont la carte de grain est prête (prepared_map)
    std::shared_ptr<std::vector<uint8_t>> prepared_map;
    int64_t net_low = 0;  // trames déjà retirées du cache du réseau (fil GPU seulement)
    std::map<int64_t, std::vector<uint8_t>> src;
    std::map<int64_t, ncnn::Mat> out;  // sorties du réseau en virgule fixe 16 bits (tampons rapatriés)
    std::map<int64_t, ncnn::Mat> refs;  // références du contre-accentuage, jusqu'au dosage
    std::map<int64_t, float> sig;
    std::map<int, restore_auto::GrainCurve> curves;
    std::map<int64_t, std::vector<float>> ana_luma;
    std::map<int64_t, std::unique_ptr<float[]>> ana_flows;
    std::map<int, std::optional<double>> ratios;
    std::optional<double> last_ratio;
    std::deque<std::vector<uint8_t>> ready;
    std::vector<std::vector<uint8_t>> spare_src, spare_out;  // tampons de trame recyclés
    std::map<int64_t, Log> logs;
    size_t held_max = 0;
    std::string err;
};
