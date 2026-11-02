/*
 * Muxiveo — interface C entre muxiveo-rife et son plugin d'inférence RIFE TensorRT (mvo-rife-trt).
 *
 * Référence unique, incluse telle quelle par mvo-rife-trt (même dépôt). Toute modification incompatible incrémente
 * MVO_TRT_ABI_VERSION ; muxiveo-rife refuse un plugin d'une autre version (repli Vulkan).
 *
 * Contrat :
 * - le plugin travaille dans le contexte CUDA primaire du périphérique `cuda_device`, celui où muxiveo-rife
 *   importe ses tampons partagés avec Vulkan ;
 * - les tenseurs sont en fp16, NCHW contigus : in0, in1, out = [1, 3, height, width] (RGB [0, 1]) ;
 * - infer() est synchrone : les écritures dans `out` sont terminées à son retour ;
 * - les chaînes sont en UTF-8 ; les fonctions ne lèvent pas d'exception.
 *
 * Interface 2 (muxiveo-rife 1.12.0, Restauration) : mvo_trt_get_api(2) rend une MvoTrtApi2, qui commence par les
 * champs de l'interface 1 et ajoute des sessions de graphe génériques (entrées in0 … in{n − 1} [1, C, H, W], sortie
 * out0 [1, C', H, W], fp16, NCHW contigus, dimensions fixées à la création). Un plugin à l'interface 2 répond aussi à
 * mvo_trt_get_api(1) (inférence RIFE inchangée).
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
    /* Ajoutés en muxiveo-rife 1.9.0 (même ABI : présents si struct_size les couvre). Identités des images in0 et in1,
     * stables tant que leur contenu ne change pas (0 : inconnue) : le plugin peut réutiliser les calculs propres à une
     * image (caractéristiques de l'encodeur), y compris entre sessions et sorties successives. */
    uint64_t id0;
    uint64_t id1;
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

#define MVO_TRT_ABI_VERSION_GRAPH 2u
#define MVO_TRT_GRAPH_MAX_INPUTS 4

typedef struct MvoTrtGraphParams
{
    uint32_t struct_size;   /* sizeof(MvoTrtGraphParams) */
    int cuda_device;        /* ordinal CUDA */
    const char* model_path; /* modèle ONNX (entrées in0…, sortie out0 ; hauteur et largeur dynamiques) */
    const char* cache_name; /* nom des moteurs en cache (unique par modèle, ex. « r2q-r2_body ») */
    const char* cache_dir;  /* moteurs et caches d'exécution (créé au besoin) */
    int width;              /* dimensions des tenseurs de la session */
    int height;
    mvo_trt_log_fn log;     /* peut être nul */
    void* log_user;
} MvoTrtGraphParams;

typedef struct MvoTrtApi2
{
    MvoTrtApi v1;           /* champs de l'interface 1 (abi_version vaut 2) ; destroy() libère aussi les sessions de graphe */
    /* Session de graphe prête (moteur construit ou relu du cache), ou NULL et raison dans `error`. */
    MvoTrtSession* (*graph_create)(const MvoTrtGraphParams* params, char* error, size_t error_size);
    /* Canaux des entrées (in_channels[0 … nombre − 1]) et de la sortie ; renvoie le nombre d'entrées, 0 si erreur. */
    int (*graph_channels)(MvoTrtSession* session, int* in_channels, int max_inputs, int* out_channels);
    /* Inférence synchrone : `inputs` (CUdeviceptr) dans l'ordre in0 …, `output` = out0 ; 1 si réussie. */
    int (*graph_infer)(MvoTrtSession* session, const uint64_t* inputs, int input_count, uint64_t output, char* error,
                       size_t error_size);
} MvoTrtApi2;

/* Point d'entrée exporté : NULL si `abi_version` n'est pas prise en charge. */
MVO_TRT_EXPORT const MvoTrtApi* mvo_trt_get_api(uint32_t abi_version);

#ifdef __cplusplus
}
#endif

#endif /* MVO_RIFE_TRT_PLUGIN_ABI_H */
