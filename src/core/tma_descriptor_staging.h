#pragma once

#include "core/device.h"

#include <cstddef>
#include <cstdint>

namespace ninfer {

// Stages an over-aligned (alignas(128)) CUtensorMap descriptor struct into device global memory
// for a kernel launch on Windows, where MSVC cannot pass the struct by value as a
// __grid_constant__ parameter (C2719). The kernel reads the descriptors from the device pointer
// returned by stage() and makes them visible to the TMA (tensormap) proxy with a
// fence.proxy.tensormap acquire before its first cp.async.bulk.tensor (see the kernel).
//
// The device destination is one persistent buffer: a per-launch cudaMallocAsync/cudaFreeAsync
// round-trip stalls the stream ~500 us per launch, while in-stream ordering guarantees the next
// launch's copy cannot start until the previous kernel has read the buffer.
//
// The host source cannot be a single buffer. cudaMemcpyAsync reads a pinned source
// asynchronously on the GPU, so the host may race ahead and overwrite it before the previous
// copy has been read - a later layer then stages its descriptors over an earlier one and the
// earlier kernel computes against the wrong tensors. The source therefore lives in a ring of
// pinned slots, each guarded by an event recorded after its copy: a slot is rewritten only
// after the copy that last used it has completed, which stalls the host solely when it laps
// the ring (i.e. runs kSlots launches ahead of the GPU).
//
// Invariants, documented here because this is the single staging site:
//   - one thread: stage() must only be called from the engine's single compute-worker thread
//     (the slot counter is not atomic);
//   - one stream: every staged launch must run on the engine's single compute stream, so the
//     persistent device buffer and the slot events order against each other in-stream;
//   - nothing is freed: the ring (one pinned allocation per slot) and the device buffer are
//     reclaimed at process exit.
template <class Descriptor, int kSlots = 32>
class TmaDescriptorStaging {
  public:
    TmaDescriptorStaging() {
        void* p = nullptr;
        CUDA_CHECK(cudaMalloc(&p, sizeof(Descriptor)));
        device_ = reinterpret_cast<Descriptor*>(p);
        for (int slot = 0; slot < kSlots; ++slot) {
            void* source = nullptr;
            CUDA_CHECK(cudaMallocHost(&source, sizeof(Descriptor)));
            host_slots_[slot] = reinterpret_cast<Descriptor*>(source);
            CUDA_CHECK(cudaEventCreateWithFlags(&slot_events_[slot], cudaEventDisableTiming));
        }
    }

    // Stages the descriptors for the next launch on `stream` and returns the device pointer
    // the kernel must read them from.
    Descriptor* stage(const Descriptor& descriptors, cudaStream_t stream) {
        const int slot = static_cast<int>(next_ % kSlots);
        if (next_ >= static_cast<std::uint64_t>(kSlots)) {
            CUDA_CHECK(cudaEventSynchronize(slot_events_[slot]));
        }
        *host_slots_[slot] = descriptors;
        CUDA_CHECK(cudaMemcpyAsync(device_, host_slots_[slot], sizeof(Descriptor),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaEventRecord(slot_events_[slot], stream));
        ++next_;
        return device_;
    }

  private:
    Descriptor* device_              = nullptr;
    Descriptor* host_slots_[kSlots]  = {};
    cudaEvent_t slot_events_[kSlots] = {};
    std::uint64_t next_              = 0;
};

} // namespace ninfer
