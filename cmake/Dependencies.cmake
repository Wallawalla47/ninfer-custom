find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# CUDA runtime is exposed as a single interface target so every component links the
# same runtime flavour. On Windows the static runtime is used so the apps do not ship
# cudart*.dll and the final process keeps a single CRT; elsewhere it is shared.
add_library(ninfer_cuda_runtime INTERFACE)
if(WIN32)
  set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
  target_link_libraries(ninfer_cuda_runtime INTERFACE CUDA::cudart_static)
else()
  set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
  target_link_libraries(ninfer_cuda_runtime INTERFACE CUDA::cudart)
endif()

# FFmpeg and hwloc: the prebuilt Windows vcpkg tree has no CMake package config for either,
# so FFmpeg is resolved from VCPKG_ROOT/VCPKG_TARGET_TRIPLET via the local find module (which
# also exposes the tree as FFMPEG_VCPKG_TREE for DLL/zlib staging) and hwloc from the same
# tree by header and import library. Windows has no io_uring: the direct read queue uses an
# I/O completion port instead (core/io/read_queue_win32.cpp). Linux uses pkg-config.
if(WIN32)
  list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
  if(DEFINED ENV{VCPKG_ROOT} AND DEFINED ENV{VCPKG_TARGET_TRIPLET})
    list(APPEND CMAKE_PREFIX_PATH
      "$ENV{VCPKG_ROOT}/installed/$ENV{VCPKG_TARGET_TRIPLET}")
  endif()
  find_package(FFMPEG REQUIRED)
  add_library(ninfer_ffmpeg_dependencies INTERFACE)
  target_include_directories(ninfer_ffmpeg_dependencies INTERFACE ${FFMPEG_INCLUDE_DIRS})
  target_link_directories(ninfer_ffmpeg_dependencies INTERFACE ${FFMPEG_LIBRARY_DIRS})
  target_link_libraries(ninfer_ffmpeg_dependencies INTERFACE ${FFMPEG_LIBRARIES})
  set(NINFER_FFMPEG_TARGET ninfer_ffmpeg_dependencies)

  find_path(NINFER_HWLOC_INCLUDE_DIR hwloc.h
    HINTS "${FFMPEG_VCPKG_TREE}/include" REQUIRED NO_DEFAULT_PATH)
  find_library(NINFER_HWLOC_LIBRARY NAMES hwloc libhwloc
    HINTS "${FFMPEG_VCPKG_TREE}/lib" REQUIRED NO_DEFAULT_PATH)
  file(STRINGS "${NINFER_HWLOC_INCLUDE_DIR}/hwloc.h" _hwloc_api
    REGEX "^#define HWLOC_API_VERSION ")
  if(NOT _hwloc_api MATCHES "0x0002(0[5-9a-f]|[1-9a-f][0-9a-f])")
    message(FATAL_ERROR "hwloc >= 2.5 is required (found ${_hwloc_api})")
  endif()
  add_library(ninfer_hwloc INTERFACE)
  target_include_directories(ninfer_hwloc INTERFACE "${NINFER_HWLOC_INCLUDE_DIR}")
  target_link_libraries(ninfer_hwloc INTERFACE "${NINFER_HWLOC_LIBRARY}")
  set(NINFER_HWLOC_TARGET ninfer_hwloc)
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(HWLOC REQUIRED IMPORTED_TARGET hwloc>=2.5)
  pkg_check_modules(URING REQUIRED IMPORTED_TARGET liburing>=2.5)
  pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat libavcodec libavutil libswscale)
  set(NINFER_FFMPEG_TARGET PkgConfig::FFMPEG)
  set(NINFER_HWLOC_TARGET PkgConfig::HWLOC)
endif()

# Repository-pinned header dependencies. No configure-time downloads.
add_library(ninfer::json INTERFACE IMPORTED GLOBAL)
target_include_directories(ninfer::json INTERFACE
  ${PROJECT_SOURCE_DIR}/third_party)

# Source base for the custom-template frontend; consumers will link it explicitly.
add_subdirectory(third_party/llama-jinja EXCLUDE_FROM_ALL)

# CPU grammar source base; consumers will link it explicitly.
add_subdirectory(third_party/xgrammar EXCLUDE_FROM_ALL)

if(NINFER_BUILD_PRODUCT_SUPPORT)
  # Media acquisition uses CURLOPT_PROTOCOLS_STR and CURLOPT_REDIR_PROTOCOLS_STR,
  # introduced in libcurl 7.85 (not merely the version of the maintainer environment).
  if(WIN32)
    find_package(CURL 7.85 REQUIRED)
    set(NINFER_CURL_TARGET CURL::libcurl)
    # The built-in FindCURL / vcpkg config exposes CURL::libcurl as an ALIAS of the
    # concrete imported target; import libs reference runtime DLLs that CMake only
    # stages when they are declared. Register them on the concrete target.
    if(TARGET CURL::libcurl_shared)
      set(_curl_real CURL::libcurl_shared)
    else()
      set(_curl_real CURL::libcurl_static)
    endif()
    if(TARGET ${_curl_real})
      file(GLOB _curl_dlls "${FFMPEG_VCPKG_TREE}/bin/*.dll")
      if(_curl_dlls)
        set_target_properties(${_curl_real} PROPERTIES
          IMPORTED_LINK_DEPENDENT_LIBRARIES "${_curl_dlls}")
      endif()
    endif()
  else()
    pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
    set(NINFER_CURL_TARGET PkgConfig::LIBCURL)
  endif()
  add_library(ninfer::httplib INTERFACE IMPORTED GLOBAL)
  target_include_directories(ninfer::httplib INTERFACE
    ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
  add_subdirectory(third_party/spdlog)
endif()
