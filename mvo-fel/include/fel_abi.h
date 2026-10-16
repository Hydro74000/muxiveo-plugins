/* SPDX-License-Identifier: LGPL-2.1-or-later */
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef _WIN32
#define MVO_FEL_EXPORT __declspec(dllexport)
#else
#define MVO_FEL_EXPORT __attribute__((visibility("default")))
#endif
#define MVO_FEL_ABI 1u
#ifdef __cplusplus
extern "C" {
#endif
typedef struct mvo_fel_context mvo_fel_context;
/* Le callback consomme le tampon avant son retour : 0 succès, sinon pipe fermé.
 * Appelé sur le thread de run ; aucun autre appel concurrent hormis cancel.
 * destroy ne doit être appelé qu'après la fin de run. Les chemins sont UTF-8. */
typedef int (*mvo_fel_write)(void *, const uint8_t *, size_t);
typedef void (*mvo_fel_progress)(void *, int64_t);
enum mvo_fel_status {
    MVO_FEL_OK=0, MVO_FEL_ERROR=1, MVO_FEL_OUTPUT_CLOSED=2,
    MVO_FEL_CANCELLED=3, MVO_FEL_INPUT_ERROR=4
};
/* INPUT_ERROR : lecture/ouverture de la source, sans reprise FEL vers BL. */
/* classification d'un NAL RPU original : -1 invalide, 0 sans EL,
 * 1 FEL, 2 MEL, 3 P7 réutilisant le mapping précédent. */
MVO_FEL_EXPORT uint32_t mvo_fel_abi_version(void);
MVO_FEL_EXPORT const char *mvo_fel_capabilities(void);
MVO_FEL_EXPORT int mvo_fel_classify(const uint8_t *nal, size_t size);
MVO_FEL_EXPORT mvo_fel_context *mvo_fel_create(const char *source, int stream,
    int threads, int fps_num, int fps_den);
/* Extensions additives ABI 1 ; disponibles si capabilities.device_selection vaut 1.
 * devices : taille nécessaire, NUL compris ; 0 en cas d'erreur d'énumération.
 * set_device avant run uniquement : "auto", "cpu" ou UUID Vulkan hexadécimal. */
MVO_FEL_EXPORT size_t mvo_fel_devices(char *buffer, size_t capacity);
/* Extension additive ABI 1 (capabilities.device_features) : même liste que mvo_fel_devices,
 * enrichie des fonctionnalités annoncées (float64 : double natif, requis par les précisions
 * exacte et standard). La liste historique garde ses clés pour les appelants antérieurs. */
MVO_FEL_EXPORT size_t mvo_fel_devices_features(char *buffer, size_t capacity);
MVO_FEL_EXPORT int mvo_fel_set_device(mvo_fel_context *, const char *device);
MVO_FEL_EXPORT const char *mvo_fel_backend(const mvo_fel_context *);
/* Extension additive ABI 1 ; disponible si capabilities.seek vaut 1. Avant run
 * uniquement : première image produite à start_us du début du fichier (sémantique
 * -ss d'entrée de FFmpeg), horodatages de sortie relatifs à ce point. */
MVO_FEL_EXPORT int mvo_fel_set_start(mvo_fel_context *, int64_t start_us);
/* Extension additive ABI 1 ; formats listés par capabilities.output_formats. Avant run :
 * gbrp16le (défaut, RGB PQ), ou Y'CbCr BT.2020nc limité 4:2:0 yuv420p16le / yuv420p10le /
 * yuv420p, sous-échantillonné au point chroma (noms FFmpeg : left, topleft...) et quantifié
 * avec le tramage déterministe demandé (none, ordered, blue ; capabilities.dithers). */
/* Extension additive ABI 1 ; modes listés par capabilities.timestamps. Avant run :
 * "source" (défaut) garde les PTS de la piste ; "zero" les rend relatifs à la première
 * image produite, comme FFmpeg le fait à l'entrée d'un NUT (alimentation directe d'un
 * encodeur sans étage FFmpeg intermédiaire). */
/* Extension additive ABI 1 ; capabilities.crop. Avant run : marges paires retirées des
 * images Y'CbCr produites (pas du RGB), la grille chroma restant celle de la source. */
MVO_FEL_EXPORT int mvo_fel_set_crop(mvo_fel_context *, int left, int top, int right, int bottom);
MVO_FEL_EXPORT int mvo_fel_set_timestamps(mvo_fel_context *, const char *mode);
/* Extension additive ABI 1 ; modes listés par capabilities.precision. Avant run, rendu GPU :
 * "fast" (défaut, float32 libplacebo), "standard" (hybride borné, ≤ 1 code 16 bits du CPU)
 * ou "exact" (double natif partout). Les deux derniers exigent un GPU calculant en double
 * (devices : float64) ; sinon le rendu passe sur CPU, qui calcule toujours en double. */
MVO_FEL_EXPORT int mvo_fel_set_precision(mvo_fel_context *, const char *mode);
MVO_FEL_EXPORT int mvo_fel_set_output(mvo_fel_context *, const char *pixel_format,
    const char *chroma_location, const char *dither);
/* Mesures locales après run ; pipe_s est inclus dans nut_s. */
MVO_FEL_EXPORT const char *mvo_fel_statistics(mvo_fel_context *);
MVO_FEL_EXPORT int mvo_fel_run(mvo_fel_context *, mvo_fel_write, mvo_fel_progress, void *);
/* Extension additive ABI 1 ; disponible si capabilities.fd_output vaut 1. Écrit le NUT
 * directement sur un descripteur POSIX ou un HANDLE Windows (pipe), sans callback ni
 * copie de l'hôte. Le descripteur reste à l'appelant. SIGPIPE est bloqué sur le thread
 * appelant pendant l'appel : un aval fermé rend MVO_FEL_OUTPUT_CLOSED. */
MVO_FEL_EXPORT int mvo_fel_run_fd(mvo_fel_context *, intptr_t fd, mvo_fel_progress, void *);
MVO_FEL_EXPORT void mvo_fel_cancel(mvo_fel_context *);
/* Erreur valide jusqu'à destroy, à consulter après run uniquement. */
MVO_FEL_EXPORT const char *mvo_fel_error(const mvo_fel_context *);
MVO_FEL_EXPORT void mvo_fel_destroy(mvo_fel_context *);
#ifdef __cplusplus
}
#endif
