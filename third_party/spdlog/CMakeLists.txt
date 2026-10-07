add_library(ninfer_third_party_spdlog STATIC
  src/async.cpp
  src/bundled_fmtlib_format.cpp
  src/cfg.cpp
  src/color_sinks.cpp
  src/file_sinks.cpp
  src/spdlog.cpp
  src/stdout_sinks.cpp)
add_library(spdlog::spdlog ALIAS ninfer_third_party_spdlog)

target_compile_features(ninfer_third_party_spdlog PUBLIC cxx_std_11)
target_compile_definitions(ninfer_third_party_spdlog PUBLIC SPDLOG_COMPILED_LIB)
target_include_directories(ninfer_third_party_spdlog SYSTEM PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ninfer_third_party_spdlog PUBLIC Threads::Threads)

# spdlog's bundled fmt enables Unicode support and static_asserts unless the compiler is
# told the source/execution charset is UTF-8. Non-UTF-8 ANSI locales (e.g. a German
# Windows install, codepage 1252) trip that assert. PUBLIC: the spdlog headers are a
# public include, so every TU that pulls them in needs the flag too.
if(MSVC)
  target_compile_options(ninfer_third_party_spdlog PUBLIC /utf-8)
endif()
