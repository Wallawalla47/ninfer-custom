if(WIN32)
  if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    message(FATAL_ERROR "NInfer CPU execution currently requires x86-64")
  endif()
  # Windows has no O_DIRECT/statx/io_uring: the file and direct read queue backends use
  # unbuffered Win32 handles and an I/O completion port behind the same headers.
  set(NINFER_IO_SOURCES io/file_win32.cpp io/read_queue_win32.cpp)
else()
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
     NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    message(FATAL_ERROR "NInfer CPU execution currently requires Linux x86-64")
  endif()
  set(NINFER_IO_SOURCES io/file.cpp io/read_queue.cpp)
endif()

add_library(ninfer_cpu STATIC
  cpu/system.cpp
  cpu/task_pool.cpp
  cpu/executor.cpp
)
ninfer_internal_includes(ninfer_cpu)
target_link_libraries(ninfer_cpu PRIVATE ${NINFER_HWLOC_TARGET} Threads::Threads)

add_library(ninfer_core STATIC
  arena.cpp
  buffer.cpp
  buffer.cu
  memory.cpp
  transfer.cpp
  transfer.cu
  device.cu
  dtype.cpp
  nf4.cpp
  ${NINFER_IO_SOURCES}
  gdn_replay_records.cpp
  cyclic_kv_cache.cpp
  host_context_arena.cpp
  host_kv_arena.cpp
  paged_kv_cache.cpp
  linear_attention_state.cpp
  state_image.cpp
  layout.cpp
  tensor.cpp
  weight_view.cpp
  weight_bank.cpp
  expert_cache.cpp
  decode_graph.cpp
)
ninfer_internal_includes(ninfer_core)
ninfer_cuda_archive(ninfer_core)
target_link_libraries(ninfer_core PUBLIC ninfer_cpu ninfer_cuda_runtime CUDA::nvtx3 CUDA::cuda_driver Threads::Threads)
if(NOT WIN32)
  target_link_libraries(ninfer_core PRIVATE PkgConfig::URING)
endif()
