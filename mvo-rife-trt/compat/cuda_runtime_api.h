/*
 * mvo-rife-trt — déclarations minimales requises par les en-têtes TensorRT for RTX, sans CUDA Toolkit.
 * Mêmes définitions que les en-têtes CUDA (pointeurs opaques) : compatibles au niveau binaire.
 */
#ifndef MVO_RIFE_TRT_CUDA_RUNTIME_API_COMPAT_H
#define MVO_RIFE_TRT_CUDA_RUNTIME_API_COMPAT_H

typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;

#endif
