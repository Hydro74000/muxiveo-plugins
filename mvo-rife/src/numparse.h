// Muxiveo — parsing numérique strict (jetons y4m, options de la ligne de commande)
// et produits entiers vérifiés.

#ifndef MUXIVEO_RIFE_NUMPARSE_H
#define MUXIVEO_RIFE_NUMPARSE_H

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>

// Entier décimal entièrement consommé, dans [min_value, max_value].
// Refuse chaîne vide, espaces, suffixes (« 2x ») et dépassements.
inline bool parse_int64_strict(const std::string& text, int64_t min_value, int64_t max_value, int64_t& out)
{
    if (text.empty() || std::isspace((unsigned char)text[0]))
        return false;
    errno = 0;
    char* end = nullptr;
    const long long value = std::strtoll(text.c_str(), &end, 10);
    if (errno == ERANGE || end != text.c_str() + text.size())
        return false;
    if (value < min_value || value > max_value)
        return false;
    out = (int64_t)value;
    return true;
}

// Réel fini entièrement consommé, dans [min_value, max_value] (NaN/infini refusés).
inline bool parse_double_strict(const std::string& text, double min_value, double max_value, double& out)
{
    if (text.empty() || std::isspace((unsigned char)text[0]))
        return false;
    errno = 0;
    char* end = nullptr;
    const double value = std::strtod(text.c_str(), &end);
    if (errno == ERANGE || end != text.c_str() + text.size() || !std::isfinite(value))
        return false;
    if (value < min_value || value > max_value)
        return false;
    out = value;
    return true;
}

// Produit de deux entiers positifs sans dépassement signé.
inline bool mul_int64_checked(int64_t a, int64_t b, int64_t& out)
{
    if (a < 0 || b < 0)
        return false;
    if (a != 0 && b > INT64_MAX / a)
        return false;
    out = a * b;
    return true;
}

#endif
