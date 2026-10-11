// Addition résiduelle et activation : même arrondi de stockage, moins de transferts.
#ifndef MUXIVEO_RESIDUAL_ADD_H
#define MUXIVEO_RESIDUAL_ADD_H
#include "layer.h"

ncnn::Layer* AddRelu_layer_creator(void* userdata);
#endif
