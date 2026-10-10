/* SPDX-License-Identifier: LGPL-2.1-or-later */
#pragma once
#include "fel_abi.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Intégration FFmpeg épinglée : pointeurs AVFrame / AVDOVIMetadata empruntés,
 * valides pendant le callback uniquement. Vérifier libavutil_version avant
 * tout échange ; cette interface n'est pas un transport inter-processus. */
typedef int (*mvo_fel_layers)(void *, const void *bl, const void *el,
    const void *metadata, int64_t pts, int64_t duration, int tb_num, int tb_den);
MVO_FEL_EXPORT unsigned mvo_fel_libavutil_version(void);
/* Retour 0 : une image ; 5 : EOF ; autres : mvo_fel_status. */
MVO_FEL_EXPORT int mvo_fel_next_layers(mvo_fel_context *, mvo_fel_layers, void *);
/* Le contexte AVHWDeviceContext Vulkan est détenu par l'appelant. */
MVO_FEL_EXPORT void *mvo_fel_vulkan_create(const void *device, int width, int height);
MVO_FEL_EXPORT int mvo_fel_vulkan_render(void *, const void *metadata,
    const void *bl, const void *el, void *output_frame);
MVO_FEL_EXPORT void mvo_fel_vulkan_destroy(void *);
#ifdef __cplusplus
}
#endif
