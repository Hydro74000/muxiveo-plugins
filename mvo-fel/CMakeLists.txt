cmake_minimum_required(VERSION 3.22)
project(mvo-fel VERSION 0.2.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(PkgConfig REQUIRED)
pkg_check_modules(AV REQUIRED IMPORTED_TARGET libavformat libavcodec libavutil)
find_package(Threads REQUIRED)
option(MVO_FEL_MATH_ONLY "Construire seulement les tests mathématiques" OFF)
option(MVO_FEL_VULKAN_PROTOTYPE "Prototype Vulkan, paquet Linux sous attestation numérique" OFF)
set(MVO_FEL_DOVI_LIBRARY "" CACHE FILEPATH "Archive libdovi statique de la révision épinglée")

# Seuils d'arrondi PQ (scripts/generate_pq_thresholds.py → data/pq_thresholds.bin) en tableau C.
set(MVO_FEL_PQ_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/pq_thresholds.c)
add_custom_command(OUTPUT ${MVO_FEL_PQ_SOURCE}
  COMMAND ${CMAKE_COMMAND} -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/data/pq_thresholds.bin
          -DOUTPUT=${MVO_FEL_PQ_SOURCE} -DSYMBOL=fel_pq_threshold_bytes
          -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_bytes.cmake
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/data/pq_thresholds.bin ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_bytes.cmake)
# Shaders GLSL de la virgule fixe (lot A2') en chaînes C.
set(MVO_FEL_GLSL_SOURCES)
foreach(shader fixed_point fixed_render)
  set(output ${CMAKE_CURRENT_BINARY_DIR}/glsl_${shader}.c)
  add_custom_command(OUTPUT ${output}
    COMMAND ${CMAKE_COMMAND} -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/${shader}.glsl
            -DOUTPUT=${output} -DSYMBOL=fel_glsl_${shader} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_text.cmake
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/${shader}.glsl ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_text.cmake)
  list(APPEND MVO_FEL_GLSL_SOURCES ${output})
endforeach()
add_library(fel_math STATIC src/reconstruction.cpp src/certified.cpp src/fixed_point.cpp ${MVO_FEL_PQ_SOURCE}
            ${MVO_FEL_GLSL_SOURCES})
target_include_directories(fel_math PUBLIC include)
target_link_libraries(fel_math PUBLIC PkgConfig::AV)
set_target_properties(fel_math PROPERTIES CXX_VISIBILITY_PRESET hidden C_VISIBILITY_PRESET hidden)
if(NOT MVO_FEL_MATH_ONLY)
  if(NOT EXISTS "${MVO_FEL_DOVI_LIBRARY}")
    message(FATAL_ERROR "Construire les dépendances avec scripts/build.py (libdovi absente)")
  endif()
  add_library(mvo_fel SHARED src/producer.cpp)
  target_compile_definitions(mvo_fel PRIVATE MVO_FEL_VERSION="${PROJECT_VERSION}")
  target_link_libraries(mvo_fel PRIVATE fel_math "${MVO_FEL_DOVI_LIBRARY}" Threads::Threads ${CMAKE_DL_LIBS})
  if(MVO_FEL_VULKAN_PROTOTYPE)
    # Le prototype reste hors du build normal ; un paquet Linux exige son
    # attestation numérique. Les autres plateformes restent à valider.
    pkg_check_modules(PLACEBO REQUIRED IMPORTED_TARGET libplacebo>=7.374.0)
    target_sources(mvo_fel PRIVATE src/gpu_backend.c src/vulkan_loader.cpp)
    target_link_libraries(mvo_fel PRIVATE PkgConfig::PLACEBO ${PLACEBO_STATIC_LDFLAGS})
    target_compile_definitions(mvo_fel PRIVATE MVO_FEL_VULKAN_PROTOTYPE=1)
  endif()
  set_target_properties(mvo_fel PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)
  if(WIN32)
    # Nom stable du manifeste, identique avec MSVC et MinGW.
    set_target_properties(mvo_fel PROPERTIES PREFIX "")
  endif()
  if(UNIX AND NOT APPLE)
    target_link_options(mvo_fel PRIVATE "-Wl,--exclude-libs,ALL"
        "-Wl,--as-needed"
        "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/exports.map")
    target_link_libraries(mvo_fel PRIVATE m)
  endif()
  if(MINGW)
    # Inclure winpthreads et les runtimes GCC ; aucune DLL MinGW à installer.
    target_link_options(mvo_fel PRIVATE -static -Wl,--exclude-libs,ALL)
    target_link_libraries(mvo_fel PRIVATE bcrypt ntdll userenv ws2_32)
  endif()
  if(NOT MSVC AND NOT APPLE)
    target_link_options(mvo_fel PRIVATE -static-libgcc -static-libstdc++)
  endif()
  if(APPLE)
    target_link_options(mvo_fel PRIVATE "-Wl,-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/src/exports.macos")
  endif()
  install(TARGETS mvo_fel LIBRARY DESTINATION . RUNTIME DESTINATION .)
endif()
if(MSVC)
  target_compile_options(fel_math PRIVATE /W4)
else()
  # Exceptions flottantes masquées : sélections sans branche, boucles PQ vectorisées.
  # Aucun changement de résultat (pas de réassociation, contrairement à -ffast-math).
  target_compile_options(fel_math PRIVATE -Wall -Wextra -Wpedantic -fno-trapping-math)
  # Reprises double-double : transformations exactes, aucune contraction en FMA.
  set_source_files_properties(src/certified.cpp PROPERTIES COMPILE_OPTIONS -ffp-contract=off)
endif()

include(CTest)
option(MVO_FEL_REFERENCE_TESTS "Comparer les pixels à libplacebo épinglé (Vulkan de test)" OFF)
if(BUILD_TESTING)
  add_executable(fel_math_test tests/reconstruction_test.cpp)
  target_link_libraries(fel_math_test PRIVATE fel_math)
  add_test(NAME fel_math COMMAND fel_math_test)
  add_executable(fel_fixed_test tests/fixed_point_test.cpp)
  target_link_libraries(fel_fixed_test PRIVATE fel_math)
  add_test(NAME fel_fixed COMMAND fel_fixed_test)
  if(NOT MVO_FEL_MATH_ONLY)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME fel_producer COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/smoke.py $<TARGET_FILE:mvo_fel>)
    add_test(NAME fel_package COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_package.py)
    add_test(NAME fel_layers COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/layers.py $<TARGET_FILE:mvo_fel>)
    add_test(NAME fel_pq_thresholds COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_pq_thresholds.py --check)
    add_executable(fel_certified_dump tests/certified_dump.cpp)
    target_link_libraries(fel_certified_dump PRIVATE fel_math)
    add_test(NAME fel_certified_eotf COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/certified_eotf.py $<TARGET_FILE:fel_certified_dump>)
  endif()
  if(MVO_FEL_REFERENCE_TESTS)
    pkg_check_modules(PLACEBO REQUIRED IMPORTED_TARGET libplacebo>=7.374.0)
    add_executable(fel_reference_test tests/reference_test.cpp tests/placebo_reference.c src/vulkan_loader.cpp)
    target_link_libraries(fel_reference_test PRIVATE fel_math PkgConfig::PLACEBO ${PLACEBO_STATIC_LDFLAGS} ${CMAKE_DL_LIBS})
    add_test(NAME fel_reference COMMAND fel_reference_test)
    add_executable(fel_gpu_reference_test tests/reference_test.cpp tests/placebo_reference.c src/gpu_backend.c src/vulkan_loader.cpp)
    target_compile_definitions(fel_gpu_reference_test PRIVATE FEL_TEST_GPU=1)
    target_link_libraries(fel_gpu_reference_test PRIVATE fel_math PkgConfig::PLACEBO ${PLACEBO_STATIC_LDFLAGS} ${CMAKE_DL_LIBS})
    add_test(NAME fel_gpu_reference COMMAND fel_gpu_reference_test)
    set_tests_properties(fel_gpu_reference PROPERTIES SKIP_RETURN_CODE 77)
    add_executable(fel_gpu_precision_test tests/gpu_precision_test.cpp src/gpu_backend.c src/vulkan_loader.cpp)
    target_link_libraries(fel_gpu_precision_test PRIVATE fel_math PkgConfig::PLACEBO ${PLACEBO_STATIC_LDFLAGS} ${CMAKE_DL_LIBS})
    add_test(NAME fel_gpu_precision COMMAND fel_gpu_precision_test)
    set_tests_properties(fel_gpu_precision PROPERTIES SKIP_RETURN_CODE 77)
    # Même test, virgule fixe imposée (lots A2' et L) : la variante doit compiler et passer l'auto-test.
    add_test(NAME fel_gpu_precision_fixed COMMAND fel_gpu_precision_test)
    set_tests_properties(fel_gpu_precision_fixed PROPERTIES SKIP_RETURN_CODE 77 ENVIRONMENT MVO_FEL_GPU_FIXED=1)
    add_executable(fel_reference_file tests/reference_file.cpp tests/placebo_reference.c src/gpu_backend.c src/vulkan_loader.cpp)
    target_link_libraries(fel_reference_file PRIVATE fel_math PkgConfig::PLACEBO ${PLACEBO_STATIC_LDFLAGS} ${CMAKE_DL_LIBS})
  endif()
endif()
