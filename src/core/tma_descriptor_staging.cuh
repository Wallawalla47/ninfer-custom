#pragma once

#include "core/device.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace ninfer {

// Stages an over-aligned (alignas(128)) CUtensorMap descriptor struct into device global memory
// for a kernel launch on Windows, where MSVC cannot pass the struct by value as a
// __grid_constant__ parameter (C2719). The kernel reads the descriptors from the device pointer
// returned by stage() and makes each tensor map visible to the TMA (tensormap) proxy with a
// fence.proxy.tensormap acquire before its first cp.async.bulk.tensor.
//
// MSVC does pass the same bytes as an 8-byte-aligned word array, so stage() launches a one-block
// kernel that takes the words by value and stores them into one persistent device buffer.
// Kernel parameters are copied when the launch is issued, and into the graph node when the
// stream is being captured, so an eager launch never reads host memory the host may since have
// rewritten, a captured launch replays exactly the descriptors it was captured with, and the
// host never waits for the GPU. The staging kernel and the TMA kernel that reads the buffer are
// ordinary stream-ordered launches, so a launch's descriptors are stored only after the previous
// TMA kernel on the stream has finished reading them.
//
// Invariants, documented here because this is the single staging site:
//   - one stream: every staged launch runs on the engine's single compute stream, so the
//     persistent device buffer orders against itself in-stream;
//   - allocation before capture: the buffer is allocated by the first stage() call, which must
//     not be inside a stream capture (an eager launch of the route precedes every capture);
//   - nothing is freed: the device buffer is reclaimed at process exit.
namespace tma_staging_detail {

template <std::size_t Words>
struct DescriptorWords {
    std::uint64_t words[Words];
};

template <std::size_t Words>
__global__ void store_descriptor_words(const DescriptorWords<Words> source,
                                       std::uint64_t* __restrict__ destination) {
    for (std::size_t word = threadIdx.x; word < Words; word += blockDim.x) {
        destination[word] = source.words[word];
    }
    // The consumer's tensor-map acquire pairs with this release.
    asm volatile("fence.proxy.tensormap::generic.release.gpu;" : : : "memory");
}

} // namespace tma_staging_detail

// Makes the tensor map at `tensor_map` (128 bytes in global memory, stored by the staging
// kernel through the generic proxy) visible to the TMA proxy. Call once per tensor map, in the
// thread that issues cp.async.bulk.tensor with it, before its first use.
__device__ __forceinline__ void acquire_staged_tensor_map(const void* tensor_map) {
    asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;"
                 :
                 : "l"(tensor_map)
                 : "memory");
}

template <class Descriptor>
class TmaDescriptorStaging {
  public:
    static_assert(sizeof(Descriptor) % sizeof(std::uint64_t) == 0);

    // Stages the descriptors for the next launch on `stream` and returns the device pointer
    // the kernel must read them from.
    Descriptor* stage(const Descriptor& descriptors, cudaStream_t stream) {
        if (device_ == nullptr) {
            void* buffer = nullptr;
            CUDA_CHECK(cudaMalloc(&buffer, sizeof(Descriptor)));
            device_ = static_cast<Descriptor*>(buffer);
        }
        tma_staging_detail::DescriptorWords<kWords> words;
        std::memcpy(words.words, &descriptors, sizeof(Descriptor));
        tma_staging_detail::store_descriptor_words<kWords>
            <<<1, 64, 0, stream>>>(words, reinterpret_cast<std::uint64_t*>(device_));
        CUDA_CHECK(cudaGetLastError());
        return device_;
    }

  private:
    static constexpr std::size_t kWords = sizeof(Descriptor) / sizeof(std::uint64_t);

    Descriptor* device_ = nullptr;
};

} // namespace ninfer
