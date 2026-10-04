# Muxiveo — muxiveo-rife : interpolation d'images RIFE v4 (Vulkan/ncnn) en flux y4m.
# Voir README.md (build) et UPSTREAM.md (provenance du code dérivé).

cmake_minimum_required(VERSION 3.20)
cmake_policy(SET CMP0091 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0091 NEW)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

project(muxiveo-rife VERSION 1.2.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(MUXIVEO_RIFE_NCNN_TAG "20260526" CACHE STRING "Tag ncnn (https://github.com/Tencent/ncnn)")
set(MUXIVEO_RIFE_NCNN_SOURCE_DIR "" CACHE PATH "Checkout ncnn local (submodule glslang inclus) ; vide = téléchargement du tag")

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# ncnn : Vulkan uniquement (chargé dynamiquement via simplevk), jeu de couches minimal
# ---------------------------------------------------------------------------
set(NCNN_VULKAN ON CACHE BOOL "" FORCE)
set(NCNN_SIMPLEVK ON CACHE BOOL "" FORCE)
set(NCNN_OPENMP OFF CACHE BOOL "" FORCE)
set(NCNN_INSTALL_SDK OFF CACHE BOOL "" FORCE)
set(NCNN_SHARED_LIB OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(NCNN_PYTHON OFF CACHE BOOL "" FORCE)
set(NCNN_PIXEL_ROTATE OFF CACHE BOOL "" FORCE)
set(NCNN_PIXEL_AFFINE OFF CACHE BOOL "" FORCE)
set(NCNN_PIXEL_DRAWING OFF CACHE BOOL "" FORCE)
set(NCNN_DISABLE_RTTI ON CACHE BOOL "" FORCE)
set(NCNN_DISABLE_EXCEPTION OFF CACHE BOOL "" FORCE)
set(NCNN_INT8 OFF CACHE BOOL "" FORCE)

# Couches nécessaires aux graphes RIFE v4 (+ dépendances internes ncnn).
set(MUXIVEO_RIFE_NCNN_LAYERS
    binaryop cast clip concat convolution convolutiondepthwise crop deconvolution
    eltwise expanddims flatten innerproduct input interp layernorm memorydata
    packing padding pixelshuffle pooling prelu relu sigmoid slice split tile unaryop
)
foreach(layer
        absval argmax batchnorm bias bnll dropout elu embed exp log lrn mvn power
        proposal reduction reshape roipooling scale softmax spp tanh threshold rnn lstm
        squeeze normalize permute priorbox detectionoutput deconvolutiondepthwise
        shufflechannel instancenorm reorg yolodetectionoutput quantize dequantize
        yolov3detectionoutput psroipooling roialign requantize hardsigmoid selu
        hardswish noop deepcopy mish statisticspooling swish gemm groupnorm softplus
        gru multiheadattention gelu convolution1d pooling1d convolutiondepthwise1d
        convolution3d convolutiondepthwise3d pooling3d matmul deconvolution1d
        deconvolutiondepthwise1d deconvolution3d deconvolutiondepthwise3d einsum
        deformableconv2d)
    set(WITH_LAYER_${layer} OFF CACHE BOOL "" FORCE)
endforeach()
foreach(layer ${MUXIVEO_RIFE_NCNN_LAYERS})
    set(WITH_LAYER_${layer} ON CACHE BOOL "" FORCE)
endforeach()

if(MUXIVEO_RIFE_NCNN_SOURCE_DIR)
    add_subdirectory(${MUXIVEO_RIFE_NCNN_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/ncnn EXCLUDE_FROM_ALL)
else()
    include(FetchContent)
    FetchContent_Declare(ncnn
        GIT_REPOSITORY https://github.com/Tencent/ncnn.git
        GIT_TAG ${MUXIVEO_RIFE_NCNN_TAG}
        GIT_SHALLOW TRUE
        GIT_SUBMODULES glslang
    )
    FetchContent_GetProperties(ncnn)
    if(NOT ncnn_POPULATED)
        FetchContent_Populate(ncnn)
        add_subdirectory(${ncnn_SOURCE_DIR} ${ncnn_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()
endif()

# ---------------------------------------------------------------------------
# Shaders : sources GLSL embarquées, compilées en SPIR-V au lancement par ncnn
# ---------------------------------------------------------------------------
set(MUXIVEO_RIFE_SHADERS
    yuv_to_rgb rgb_to_yuv pack_samples rife_v4_timestep warp warp_pack4
    tta_flip tta_accumulate tta_resolve
)
set(SHADER_HEADERS)
foreach(shader ${MUXIVEO_RIFE_SHADERS})
    set(src ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/${shader}.comp)
    set(hdr ${CMAKE_CURRENT_BINARY_DIR}/${shader}.comp.hex.h)
    add_custom_command(
        OUTPUT ${hdr}
        COMMAND ${CMAKE_COMMAND} -DSHADER_SRC=${src} -DSHADER_COMP_HEADER=${hdr}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/generate_shader_comp_header.cmake
        DEPENDS ${src} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/generate_shader_comp_header.cmake
        COMMENT "Shader ${shader}.comp"
        VERBATIM
    )
    list(APPEND SHADER_HEADERS ${hdr})
endforeach()

add_executable(muxiveo-rife
    src/main.cpp
    src/engine.cpp
    src/y4m.cpp
    src/warp.cpp
    src/uhd.cpp
    ${SHADER_HEADERS}
)
target_include_directories(muxiveo-rife PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(muxiveo-rife PRIVATE
    MUXIVEO_RIFE_VERSION="${PROJECT_VERSION}"
    MUXIVEO_RIFE_NCNN_VERSION="${MUXIVEO_RIFE_NCNN_TAG}"
)
target_link_libraries(muxiveo-rife PRIVATE ncnn Threads::Threads)

if(WIN32)
    target_compile_definitions(muxiveo-rife PRIVATE UNICODE _UNICODE NOMINMAX _CRT_SECURE_NO_WARNINGS)
    target_link_libraries(muxiveo-rife PRIVATE shell32)
elseif(NOT APPLE)
    # binaire portable : runtime C++ statique (si libstdc++.a disponible), glibc dynamique
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_LINK_OPTIONS -static-libstdc++ -static-libgcc)
    check_cxx_source_compiles("#include <string>\nint main(){std::string s(\"x\");return (int)s.size()-1;}" MUXIVEO_RIFE_HAS_STATIC_LIBSTDCXX)
    unset(CMAKE_REQUIRED_LINK_OPTIONS)
    if(MUXIVEO_RIFE_HAS_STATIC_LIBSTDCXX)
        target_link_options(muxiveo-rife PRIVATE -static-libstdc++ -static-libgcc)
    else()
        message(WARNING "libstdc++ statique indisponible : binaire lié dynamiquement (non portable)")
    endif()
endif()

install(TARGETS muxiveo-rife RUNTIME DESTINATION .)
