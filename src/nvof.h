// Muxiveo — flux optique matériel NVIDIA (NVOFA), facultatif.
// Pilote chargé dynamiquement (libnvidia-opticalflow.so.1 + libcuda.so.1, nvofapi64.dll +
// nvcuda.dll sous Windows) : aucune dépendance à la compilation ni au lancement. Sans carte
// NVIDIA ou sans pilote compatible, init() échoue et le moteur reste purement Vulkan.

#ifndef MUXIVEO_RIFE_NVOF_H
#define MUXIVEO_RIFE_NVOF_H

#include <cstdint>
#include <string>
#include <vector>

class NvofFlow
{
public:
    NvofFlow();
    ~NvofFlow();

    // device_name / rank : GPU Vulkan choisi (appariement CUDA par nom, puis rang parmi les homonymes).
    // Flux calculé sur la luma réduite 2×2 avec une grille de 2 px (4 px en pleine résolution :
    // même qualité, charge CPU et bus réduite) ; repli en pleine résolution avec une grille de
    // 4 px sur les GPU qui n'offrent que cette grille (Turing). Niveau de qualité SLOW.
    bool init(int width, int height, const std::string& device_name, int rank, std::string& error);
    int scale() const { return scale_; }
    int width() const { return w; }
    int height() const { return h; }
    bool active() const { return ready; }
    const std::string& device() const { return cuda_name; }

    // Flux aller (y0 → y1) et retour (y1 → y0), luma 8 bits réduite (pas = width()),
    // vecteurs S10.5 entrelacés (x, y) sur la grille grid_w() × grid_h().
    bool compute(const uint8_t* y0, const uint8_t* y1, std::vector<int16_t>& fwd, std::vector<int16_t>& bwd,
                 std::string& error);
    int grid() const { return grid_; }   // en pixels de l'image analysée
    int grid_w() const { return (w + grid() - 1) / grid(); }
    int grid_h() const { return (h + grid() - 1) / grid(); }

private:
    struct Impl;
    Impl* d;
    bool ready;
    int w;
    int h;
    int scale_ = 2;
    int grid_ = 2;
    std::string cuda_name;
};

#endif // MUXIVEO_RIFE_NVOF_H
