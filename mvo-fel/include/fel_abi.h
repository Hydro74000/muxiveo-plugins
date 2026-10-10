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
MVO_FEL_EXPORT int mvo_fel_set_device(mvo_fel_context *, const char *device);
MVO_FEL_EXPORT const char *mvo_fel_backend(const mvo_fel_context *);
/* Mesures locales après run ; pipe_s est inclus dans nut_s. */
MVO_FEL_EXPORT const char *mvo_fel_statistics(mvo_fel_context *);
MVO_FEL_EXPORT int mvo_fel_run(mvo_fel_context *, mvo_fel_write, mvo_fel_progress, void *);
MVO_FEL_EXPORT void mvo_fel_cancel(mvo_fel_context *);
/* Erreur valide jusqu'à destroy, à consulter après run uniquement. */
MVO_FEL_EXPORT const char *mvo_fel_error(const mvo_fel_context *);
MVO_FEL_EXPORT void mvo_fel_destroy(mvo_fel_context *);
#ifdef __cplusplus
}
#endif
