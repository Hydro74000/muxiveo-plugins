// Restauration (R2) : mesures de l'Auto et dosage, calculés en flux sur la luma 10 bits (processeur).
//
// Portage de la référence du banc (r1r_auto_stream.py, Muxiveo, plan Restauration § 5 bis, jalon P1) :
// - coupes de plan en flux : écart moyen de la luma au quart entre images consécutives, coupe si l'écart dépasse
//   max(4 × médiane, 8) ; médiane sur la fenêtre [t − past, t + look] (aucune passe préalable sur le film) ;
// - courbe de grain σ(luminance) d'une image et carte de grain (entrée du réseau), courbe du plan = médiane des courbes ;
// - rapport de grain d'un plan face au profil cible : résidu face à la médiane recalée t ± 2 par les flux du réseau,
//   sur les aplats où l'aller-retour des flux tient ; formule bornée contre le surnettoyage ;
// - σ du résidu (source − sortie) pour l'égalisation du dosage par plan ;
// - dosage : sortie + part × égalisation × (source − sortie), un seul arrondi, tramage triangulaire en option.
//
// Conventions : luma en codes 10 bits ; flous gaussiens et Sobel comme OpenCV (taille de noyau d'après σ, bords en
// miroir sans répétition du bord) ; centiles à interpolation linéaire (NumPy) ; recalage bilinéaire, bords répliqués,
// fraction au 1/32 de pixel (cv2.remap).
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <vector>

namespace restore_auto
{

constexpr int GRAIN_BINS = 14;  // tranches de luma : bornes linspace(64, 940, 15)

// Borne i (0…14) des tranches de luma.
double bin_edge(int i);

struct GrainCurve
{
    std::array<float, GRAIN_BINS> center{};
    std::array<float, GRAIN_BINS> sigma{};
};

// Courbe σ(luminance) d'une image (r0_eval.grain_sigma) : passe-haut 3 × 3 sur les aplats (gradient de la luma
// moyennée 5 × 5 sous son 30e centile), écart type robuste (p84 − p16) / 2 par tranche de plus de 3000 pixels ; tranche
// vide : valeur de la tranche mesurée la plus proche ; aucune tranche mesurée : 0,5 partout.
GrainCurve grain_curve(const uint16_t* y, int w, int h, std::ptrdiff_t stride);

// Courbe d'un plan : médiane, tranche par tranche, des courbes de ses images (r1_nmap_shot.shot_curve).
GrainCurve median_curve(const std::vector<GrainCurve>& curves);

// Carte de grain (octet = σ × 20, borné à 255) : courbe interpolée sur la luma lissée (gaussienne σ 2), r1_nmap_shot.map_from_curve.
void grain_map(const uint16_t* y, int w, int h, std::ptrdiff_t stride, const GrainCurve& curve, uint8_t* map);

// σ robuste (1,4826 × MAD, au moins 1e-3) du passe-haut (x − gaussienne σ 1,2) de la luma du résidu source − sortie.
float residual_sigma(const uint16_t* src_y, std::ptrdiff_t src_stride, const float* out_y, std::ptrdiff_t out_stride, int w, int h);

// Coupes de plan en flux. push() une image après l'autre, finish() à la fin du flux ; decided() = dernière image dont
// la coupe est décidée (une coupe en t est décidée dès que l'écart de t + look est connu, ou à la fin).
class CutDetector
{
public:
    explicit CutDetector(int past = 48, int look = 12, double factor = 4.0, double floor = 8.0);
    void push(const uint16_t* y, int w, int h, std::ptrdiff_t stride);
    void finish();
    int frames() const { return frames_; }
    int decided() const { return decided_; }
    bool is_cut(int t) const { return t >= 0 && t < (int)cut_.size() && cut_[t]; }
    // Début du plan de l'image t (t ≤ decided()).
    int shot_start(int t) const;

private:
    void decide();
    int past_, look_;
    double factor_, floor_;
    int frames_ = 0, decided_ = 0;
    bool finished_ = false;
    std::vector<float> prev_;
    std::vector<float> diff_;      // diff_[t] : écart moyen entre t − 1 et t (diff_[0] inutilisé)
    std::vector<uint8_t> cut_;     // cut_[t] : t commence un plan
};

// Image de mesure du rapport : luma recadrée (h × w, codes 10 bits en flottant) et flux du réseau vers t − 2, t − 1,
// t + 1, t + 2 sur le même recadrage (4 plans h × w × 2 entrelacés, en pixels).
struct AnalysisFrame
{
    const float* luma = nullptr;
    const float* flows = nullptr;
};

struct GrainBins
{
    std::array<std::optional<double>, GRAIN_BINS> sigma;  // σ médian des centres (tranche mesurée)
    std::array<double, GRAIN_BINS> pixels{};              // pixels retenus, sommés sur les centres
};

// σ du grain par tranche (r0_real.measure, partie σ) sur les centres donnés : résidu face à la médiane des 5 images
// recalées, aplats (gradient de la médiane, lissé σ 2, sous son 40e centile) où l'aller-retour des 4 flux reste sous
// `ar` px, tranches de plus de 4000 pixels, × biais de la médiane de 5 ; médiane des centres. Les images c ± 1, c ± 2
// de chaque centre c doivent être présentes.
GrainBins sigma_bins(const std::map<int, AnalysisFrame>& frames, const std::vector<int>& centers, int w, int h, double ar = 1.0);

// Biais des flux du réseau face au flux optique du banc (σ surestimé par le désalignement), selon la largeur de l'image :
// 0,91 jusqu'à 1920 px, 0,76 à 3840 px, linéaire entre les deux.
double flow_bias(int width);

// Rapport de grain du plan face au profil cible : min(A, 2 × B) × flow_bias(largeur), plafonné à 10 ; A = moyenne des σ /
// moyenne du profil sur les tranches communes (Auto v2), B = médiane, pondérée par les pixels, des σ / profil par
// tranche. A saute quand une tranche claire (profil 0,22–0,37) entre dans le masque ; B le borne. Sans tranche commune :
// σ moyen face au niveau médian du profil ; aucune mesure : vide.
std::optional<double> ratio_from_sigma(const GrainBins& bins, int width);

// Niveau de l'Auto (Fort 0,5, Auto / Moyen 1, Léger 1,5) ; contenu animé : un cran vers Fort (k × 0,5).
inline double auto_k(double k, bool animation) { return animation ? k * 0.5 : k; }

// Part du résidu réinjectée : min(1, k / rapport) ; rapport inconnu : 0,25 × k.
double auto_part(std::optional<double> ratio, double k);

// Dosage d'un plan (codes planaires, n valeurs) : dst = arrondi(sortie + kk × (source − sortie)) borné à [lo, hi].
// `dither_frame` ≥ 0 : arrondi tramé (bruit triangulaire ±1 pas, déterministe d'après le numéro d'image et la position) ;
// `dither_step` : pas du tramage en codes (1 ; 4 = pas 8 bits d'une sortie 10 bits destinée à un encodeur 8 bits : codes
// multiples de 4, conversion exacte en aval).
void dose(const uint16_t* src, const float* out, std::size_t n, double kk, uint16_t lo, uint16_t hi, uint16_t* dst,
          long long dither_frame = -1, std::size_t dither_offset = 0, int dither_step = 1);

// Fils processeur des calculs par image (0 : selon le processeur ; résultats identiques quel que soit le nombre).
void set_threads(int n);

// Outils exposés pour les tests.
namespace detail
{
// Exécute fn(début, fin) sur des bandes de [0, n), en parallèle (fils de set_threads()).
void for_rows(int n, const std::function<void(int, int)>& fn);
double percentile(std::vector<float> v, double q);  // q en % (interpolation linéaire, NumPy)
int gaussian_ksize(double sigma);                   // taille de noyau d'OpenCV pour une image flottante
void gaussian_blur(const float* src, int w, int h, double sigma, float* dst);
void area_resize(const float* src, int w, int h, int dw, int dh, float* dst);
}  // namespace detail

}  // namespace restore_auto
