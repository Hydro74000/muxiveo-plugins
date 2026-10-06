// Muxiveo — lecture / écriture de flux YUV4MPEG2 (y4m) en streaming.

#ifndef MUXIVEO_RIFE_Y4M_H
#define MUXIVEO_RIFE_Y4M_H

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// Position des échantillons de chroma (ITU-T H.273 chroma_sample_loc_type).
enum class ChromaSiting
{
    Unknown,
    Left,     // type 0 : MPEG-2, co-situé horizontalement, centré verticalement
    Center,   // type 1 : JPEG / MPEG-1, centré dans les deux axes
    TopLeft,  // type 2 : co-situé dans les deux axes (UHD BD, HDR10 typique)
};

enum class ColorRange
{
    Unknown,
    Limited,
    Full,
};

// Bornes des en-têtes acceptés : tailles calculées sans dépassement en 64 bits
// et allocations raisonnables (refus explicite au-delà).
constexpr int64_t Y4M_MAX_DIMENSION = 32768;
constexpr int64_t Y4M_MAX_LUMA_SAMPLES = int64_t(1) << 28; // 16384 × 16384
constexpr int64_t Y4M_MAX_RATE_TERM = 0x7FFFFFFF;          // numérateur/dénominateur de cadence

struct FrameFormat
{
    int width = 0;
    int height = 0;
    int sub_x = 2;           // facteur de sous-échantillonnage chroma horizontal (1 ou 2)
    int sub_y = 2;           // facteur de sous-échantillonnage chroma vertical (1 ou 2)
    int bit_depth = 8;
    int bytes_per_sample = 1;
    int64_t fps_num = 0;
    int64_t fps_den = 1;
    char interlace = 'p';    // 'p', 't', 'b', 'm' ou '?'
    ChromaSiting siting = ChromaSiting::Unknown;
    ColorRange range = ColorRange::Unknown;

    // Calcul en 64 bits : dimensions bornées par Y4M_MAX_DIMENSION à la lecture.
    int chroma_width() const { return (int)(((int64_t)width + sub_x - 1) / sub_x); }
    int chroma_height() const { return (int)(((int64_t)height + sub_y - 1) / sub_y); }
    int64_t luma_samples() const { return (int64_t)width * height; }
    int64_t chroma_samples() const { return (int64_t)chroma_width() * chroma_height(); }
    int64_t total_samples() const { return luma_samples() + 2 * chroma_samples(); }
    size_t frame_bytes() const { return (size_t)(total_samples() * bytes_per_sample); }
    // taille arrondie au mot 32 bits (transferts GPU)
    size_t padded_frame_bytes() const { return (frame_bytes() + 3) / 4 * 4; }
};

class Y4mReader
{
public:
    explicit Y4mReader(FILE* fp);

    // Lit l'en-tête. Retourne false et remplit `error` en cas d'en-tête invalide/non supporté.
    bool read_header(std::string& error);

    // Lit la trame suivante dans `dst` (taille >= frame_bytes()).
    // Retourne 1 si une trame a été lue, 0 en fin de flux, -1 en cas d'erreur.
    int read_frame(uint8_t* dst, std::string& error);

    const FrameFormat& format() const { return fmt; }

    // Jetons d'en-tête autres que W/H/F (réémis tels quels en sortie).
    const std::vector<std::string>& passthrough_tokens() const { return tokens; }

private:
    bool read_line(std::string& line, size_t max_len);

    FILE* fp;
    FrameFormat fmt;
    std::vector<std::string> tokens;
};

class Y4mWriter
{
public:
    explicit Y4mWriter(FILE* fp);

    bool write_header(const FrameFormat& fmt, const std::vector<std::string>& passthrough_tokens);
    bool write_frame(const uint8_t* data, size_t size);
    bool flush();

private:
    FILE* fp;
};

#endif // MUXIVEO_RIFE_Y4M_H
