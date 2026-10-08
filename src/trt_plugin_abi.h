/*
 * Muxiveo — interface C entre muxiveo-rife et son plugin d'inférence RIFE TensorRT (mvo-rife-trt).
 *
 * Référence : native/muxiveo-rife/src/trt_plugin_abi.h du dépôt Muxiveo (MIT). Le dépôt muxiveo-plugins en
 * garde une copie identique, vérifiée par sa CI. Toute modification incompatible incrémente
 * MVO_TRT_ABI_VERSION ; muxiveo-rife refuse un plugin d'une autre version (repli Vulkan).
 *
 * Contrat :
 * - le plugin travaille dans le contexte CUDA primaire du périphérique `cuda_device`, celui où muxiveo-rife
 *   importe ses tampons partagés avec Vulkan ;
 * - les tenseurs sont en fp16, NCHW contigus : in0, in1, out = [1, 3, height, width] (RGB [0, 1]) ;
 * - infer() est synchrone : les écritures dans `out` sont terminées à son retour ;
 * - les chaînes sont en UTF-8 ; les fonctions ne lèvent pas d'exception.
 */

#ifndef MVO_RIFE_TRT_PLUGIN_ABI_H
#define MVO_RIFE_TRT_PLUGIN_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MVO_TRT_ABI_VERSION 1u

#if defined(_WIN32)
#define MVO_TRT_EXPORT __declspec(dllexport)
#else
#define MVO_TRT_EXPORT __attribute__((visibility("default")))
#endif

/* Niveaux de journal transmis à muxiveo-rife (lignes « info: » / « warning: »). */
#define MVO_TRT_LOG_INFO 0
#define MVO_TRT_LOG_WARNING 1

typedef void (*mvo_trt_log_fn)(void* user, int level, const char* message);

typedef struct MvoTrtSession MvoTrtSession;

typedef struct MvoTrtCreateParams
{
    uint32_t struct_size;   /* sizeof(MvoTrtCreateParams) */
    int cuda_device;        /* ordinal CUDA */
    const char* model_path; /* modèle ONNX du plugin */
    const char* cache_dir;  /* moteurs et caches d'exécution (créé au besoin) */
    int width;              /* dimensions paddées des tenseurs */
    int height;
    mvo_trt_log_fn log;     /* peut être nul */
    void* log_user;
} MvoTrtCreateParams;

typedef struct MvoTrtInferParams
{
    uint32_t struct_size;   /* sizeof(MvoTrtInferParams) */
    uint64_t in0;           /* CUdeviceptr */
    uint64_t in1;
    uint64_t out;
    float t;                /* position temporelle (0 < t < 1) */
} MvoTrtInferParams;

typedef struct MvoTrtApi
{
    uint32_t abi_version;
    /* Version du plugin (ex. « 1.0.0 (TensorRT-RTX 1.6.1) »). */
    const char* (*version)(void);
    /* 1 si le périphérique peut exécuter le plugin, sinon 0 et raison dans `error`. */
    int (*device_supported)(int cuda_device, char* error, size_t error_size);
    /* Session prête à l'inférence (moteur construit ou relu du cache), ou NULL et raison dans `error`. */
    MvoTrtSession* (*create)(const MvoTrtCreateParams* params, char* error, size_t error_size);
    /* 1 si l'image est produite, sinon 0 et raison dans `error`. */
    int (*infer)(MvoTrtSession* session, const MvoTrtInferParams* params, char* error, size_t error_size);
    void (*destroy)(MvoTrtSession* session);
} MvoTrtApi;

/* Point d'entrée exporté : NULL si `abi_version` n'est pas prise en charge. */
MVO_TRT_EXPORT const MvoTrtApi* mvo_trt_get_api(uint32_t abi_version);

#ifdef __cplusplus
}
#endif

#endif /* MVO_RIFE_TRT_PLUGIN_ABI_H */
