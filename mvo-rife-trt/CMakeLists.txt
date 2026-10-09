# mvo-rife-trt — plugin d'inférence RIFE TensorRT for RTX pour muxiveo-rife.
cmake_minimum_required(VERSION 3.20)
project(mvo-rife-trt VERSION 1.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(TRTRTX_ROOT "" CACHE PATH "SDK TensorRT for RTX extrait (include/, lib/)")
if(NOT TRTRTX_ROOT OR NOT EXISTS "${TRTRTX_ROOT}/include/NvInfer.h")
    message(FATAL_ERROR "TRTRTX_ROOT doit désigner le SDK TensorRT for RTX extrait (include/NvInfer.h introuvable)")
endif()
find_library(TRTRTX_LIB NAMES tensorrt_rtx tensorrt_rtx_1_6 PATHS "${TRTRTX_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(TRTRTX_ONNX_LIB NAMES tensorrt_onnxparser_rtx tensorrt_onnxparser_rtx_1_6 PATHS "${TRTRTX_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)

add_library(mvo_rife_trt SHARED src/plugin.cpp)
# compat/ : types CUDA minimaux requis par les en-têtes TensorRT (pas de CUDA Toolkit).
# Interface avec muxiveo-rife : en-tête de référence du moteur (mvo-rife/src), jamais recopié.
target_include_directories(mvo_rife_trt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../mvo-rife/src compat)
# en-têtes NVIDIA en « système » : leurs avertissements d'obsolescence ne concernent pas ce code
target_include_directories(mvo_rife_trt SYSTEM PRIVATE "${TRTRTX_ROOT}/include")
target_compile_definitions(mvo_rife_trt PRIVATE MVO_RIFE_TRT_VERSION="${PROJECT_VERSION}")
target_link_libraries(mvo_rife_trt PRIVATE "${TRTRTX_LIB}" "${TRTRTX_ONNX_LIB}" ${CMAKE_DL_LIBS})
set_target_properties(mvo_rife_trt PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    # TensorRT for RTX livré à côté du plugin
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "$ORIGIN")
if(WIN32)
    target_compile_definitions(mvo_rife_trt PRIVATE UNICODE _UNICODE NOMINMAX)
    # runtime MSVC statique, comme TensorRT for RTX : aucune DLL Visual C++ à livrer
    set_property(TARGET mvo_rife_trt PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
    # libstdc++ dynamique : TensorRT for RTX l'exige déjà ; une copie statique dans le plugin intercepterait ses
    # symboles au chargement (plantage). Compilé en manylinux_2_28 (glibc >= 2.28).
    target_link_options(mvo_rife_trt PRIVATE -Wl,--no-undefined $<$<CONFIG:Release>:-s>)
endif()
