// Restauration (R2) : étage en flux, placé avant l'interpolation. Portage de StreamAuto (r1r_auto_stream.py, banc,
// plan Restauration § 5 bis, jalon P2) : chaque trame est lue une fois, dans l'ordre ; chaque décision n'utilise qu'une
// fenêtre bornée, aucune passe préalable sur le film.
//
// Étapes, chacune dès que sa fenêtre est complète : écart de luma au quart → décision de coupe (t + cut_look) → courbe
// de grain du plan (images [s, s + look)) → réseau R2 de l'image u (coupes décidées jusqu'à u + 2 ; voisines du même
// plan, miroir aux bords et aux coupes) → σ du résidu → rapport de grain du plan (flux du réseau des images s … s + 10,
// formule bornée, biais selon la largeur) → dosage de t (égalisation sur [t − dose_look, t + dose_look] ∩ plan, un seul
// arrondi, tramage). Trames de sortie au format d'entrée, dans l'ordre.
#pragma once
#include <cstdint>
#include <deque>
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
    // net : réseau configuré pour le format ; fmt : format des trames (8 à 16 bits, 4:2:0, 4:2:2 ou 4:4:4).
    bool init(RestoreNet* net, const FrameFormat& fmt, const ColorParams& color, const RestoreOptions& opt, std::string& error);
    // Trame brute (frame_bytes octets, tampon recopié). Faux en cas d'erreur (error()).
    bool push(const uint8_t* frame);
    // Fin du flux : toutes les trames restantes deviennent disponibles.
    bool finish();
    // Prochaine trame restaurée (format d'entrée), si disponible.
    bool pop(std::vector<uint8_t>& frame);
    const std::string& error() const { return err; }
    // Diagnostic : trames tenues (sources + sorties du réseau) au plus.
    size_t max_held() const { return held_max; }
    // Journal : rapport et part réinjectée par image (premier niveau).
    struct Log
    {
        std::optional<double> ratio;
        double part = 0.0, k_eff = 0.0;
    };
    const std::map<int64_t, Log>& log() const { return logs; }
    const std::vector<int>& cuts() const { return cut_list; }

private:
    bool pump();
    bool run_net();
    bool emit();
    bool shot_curve(int s);
    bool shot_ratio(int s);
    int shot_start(int t) const { return cuts_.shot_start(t); }
    std::optional<int> shot_end(int s) const;
    bool same_shot(int t, int u) const;
    int neighbour(int t, int d) const;
    std::vector<uint16_t> luma10(const std::vector<uint8_t>& frame) const;
    void samples(const std::vector<uint8_t>& frame, std::vector<uint16_t>& out) const;
    void release_before(int64_t keep_src, int64_t keep_out);

    RestoreNet* net = nullptr;
    FrameFormat fmt;
    ColorParams color;
    RestoreOptions o;
    double k = 1.0;
    uint16_t lo = 0, hi = 0;
    RestoreNet::Crop crop;
    restore_auto::CutDetector cuts_;
    std::vector<int> cut_list;
    int cuts_scanned = 0;  // trames déjà examinées pour cut_list
    std::optional<int> n_total;
    int n_in = 0, next_net = 0, next_out = 0;
    std::map<int64_t, std::vector<uint8_t>> src;                 // trames sources tenues
    std::map<int64_t, std::vector<float>> out;                   // sorties du réseau (codes flottants)
    std::map<int64_t, float> sig;                                // σ du résidu
    std::map<int, restore_auto::GrainCurve> curves;              // courbe de grain par début de plan
    std::map<int64_t, std::vector<float>> ana_luma, ana_flows;   // images de mesure du rapport (recadrage)
    std::map<int, std::optional<double>> ratios;
    std::optional<double> last_ratio;
    std::deque<std::vector<uint8_t>> ready;
    std::map<int64_t, Log> logs;
    size_t held_max = 0;
    std::string err;
};
