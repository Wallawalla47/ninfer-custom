// A focused regression test for the Windows-only NVFP4 W4A4 TMA descriptor-staging race.
//
// The Windows staging path stores the over-aligned CUtensorMap descriptor struct into one
// persistent device buffer that every eager launch reuses, and gives each launch captured into a
// CUDA Graph a device copy of its own (core/tma_descriptor_staging.cuh). A launch that reads
// another launch's descriptors (staged too early, too late, or a stale tensor map in the TMA
// proxy) computes against a foreign weight. This harness fires the TMA route
// back-to-back (no inter-launch sync) with distinct weights and compares every launch against an
// isolated (synced) reference: any divergence means a launch read a stale or foreign descriptor.
// It covers both staging sites: the plain Linear TMA (launch_tma) and the fused LinearSwiGLU TMA.

#include "core/arena.h"
#include "core/buffer.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "cuda_fixture.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string_view>
#include <utility>
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::test;

Device device() { return cuda_environment().context.device(); }

quantized_weight::PackedWeight make_nvfp4(std::int32_t n, std::int32_t k, std::uint32_t seed) {
    quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor  = 3.5F;
    return quantized_weight::make_patterned_weight(QType::NVFP4, n, k, seed, options);
}

std::vector<std::uint16_t> make_activation(std::int32_t k, std::int32_t tokens,
                                           std::uint32_t seed) {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(k) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t column = 0; column < k; ++column) {
            std::uint32_t coordinate =
                seed ^ static_cast<std::uint32_t>(column) * 0x9e3779b9U ^
                static_cast<std::uint32_t>(token) * 0x85ebca6bU;
            coordinate ^= coordinate >> 16;
            coordinate *= 0x7feb352dU;
            coordinate ^= coordinate >> 15;
            const int raw     = static_cast<int>(coordinate & 0xffU);
            const float value = static_cast<float>(raw - 128) * (1.0F / 256.0F);
            result[static_cast<std::size_t>(token) * k + column] = f32_to_bf16(value);
        }
    }
    return result;
}

// Fire `launches` isolated launches (synchronizing after each) to obtain per-weight reference
// outputs, then fire `launches` back-to-back launches (no inter-launch sync), then replay each
// launch from a captured CUDA Graph, and bit-exact compare both against the references.
// `launch(i, output, execution)` enqueues exactly one op call for weight i into output. The
// kernels are deterministic for fixed inputs, so a correct run reproduces the reference exactly;
// a staging race feeds a launch a foreign weight and the outputs diverge.
template <typename Launch>
int run_staging_race(std::string_view label, std::int32_t launches, std::int32_t tokens,
                     std::size_t output_elements, Launch launch) {
    const std::size_t output_bytes = output_elements * sizeof(std::uint16_t);
    std::vector<DeviceBuffer> reference, raced, replayed;
    for (std::int32_t i = 0; i < launches; ++i) {
        reference.emplace_back(device(), output_bytes);
        raced.emplace_back(device(), output_bytes);
        replayed.emplace_back(device(), output_bytes);
    }
    DeviceBuffer scratch(device(), output_bytes);
    const auto eager = cuda_environment().default_execution;
    cuda_check(cudaStreamSynchronize(eager.stream()), "idle before reference phase");
    for (std::int32_t i = 0; i < launches; ++i) {
        launch(i, reference[i].span(), eager);
        cuda_check(cudaStreamSynchronize(eager.stream()), "synchronize reference launch");
    }
    // Back-to-back: the host enqueues every launch before the GPU has run the earlier ones, so
    // staging that reads host memory after the host moved on feeds a launch foreign descriptors.
    for (std::int32_t i = 0; i < launches; ++i) launch(i, raced[i].span(), eager);
    cuda_check(cudaStreamSynchronize(eager.stream()), "synchronize raced launches");

    // Captured: each launch is captured into its own graph, more eager launches than any
    // host-side staging state holds follow, and every graph then replays right after a foreign
    // eager launch has restaged the shared descriptor buffer. A replay must use the descriptors
    // it was captured with, whatever the host and the eager buffer held since.
    DeviceContext capture_context;
    const auto captured = capture_context.execution_view();
    const auto stream   = captured.stream();
    std::vector<DecodeGraphDefinition> definitions(launches);
    std::vector<DecodeGraphExecutable> graphs(launches);
    for (std::int32_t i = 0; i < launches; ++i) {
        definitions[i].capture(stream, [&] { launch(i, replayed[i].span(), captured); });
        graphs[i].instantiate(definitions[i]);
    }
    constexpr std::int32_t kRestagings = 48;
    for (std::int32_t round = 0; round < kRestagings; ++round)
        launch(round % launches, scratch.span(), captured);
    for (std::int32_t i = 0; i < launches; ++i) {
        launch((i + 1) % launches, scratch.span(), captured);
        graphs[i].launch(stream);
    }
    cuda_check(cudaStreamSynchronize(stream), "synchronize graph replays");

    std::vector<std::uint16_t> reference_bits(output_elements), bits(output_elements);
    const auto divergence = [&](const DeviceBuffer& output) {
        output.copy_to_host(bits.data(), output_bytes);
        long long divergent = 0;
        for (std::size_t j = 0; j < output_elements; ++j) {
            if (bits[j] != reference_bits[j]) { ++divergent; }
        }
        return divergent;
    };
    long long total_divergent = 0;
    for (std::int32_t i = 0; i < launches; ++i) {
        reference[i].copy_to_host(reference_bits.data(), output_bytes);
        for (const auto& [phase, output] :
             {std::pair<std::string_view, const DeviceBuffer*>{"back-to-back", &raced[i]},
              std::pair<std::string_view, const DeviceBuffer*>{"graph replay", &replayed[i]}}) {
            const long long divergent = divergence(*output);
            if (divergent != 0) {
                std::cerr << label << " T=" << tokens << " launch=" << i << " (" << phase
                          << "): " << divergent << '/' << output_elements
                          << " output elements diverged from the isolated reference\n";
            }
            total_divergent += divergent;
        }
    }
    return total_divergent == 0 ? 0 : 1;
}

struct Weights {
    std::vector<quantized_weight::PackedWeight> host;
    std::vector<DeviceBuffer> device;

    Weights(std::int32_t count, std::int32_t n, std::int32_t k, std::uint32_t seed) {
        for (std::int32_t i = 0; i < count; ++i) {
            host.push_back(make_nvfp4(n, k, seed + static_cast<std::uint32_t>(i)));
            device.emplace_back(::ninfer::test::cuda_environment().context.device(),
                                host.back().payload.size());
            device.back().copy_from_host(host.back().payload.data(), device.back().size());
        }
    }

    [[nodiscard]] Weight weight(std::int32_t i) const {
        return host[i].device_weight(device[i].span());
    }
};

int run_linear_race() {
    constexpr std::int32_t kN      = 5120;
    constexpr std::int32_t kK      = 6144; // N5120K6144: the smallest registered TMA problem
    constexpr std::int32_t kTokens = 1024; // this shape takes the TMA route from T=1024
    constexpr std::int32_t kLaunch = 8;
    const Weights weights(kLaunch, kN, kK, 723U);
    const auto activation_bits = make_activation(kK, kTokens, 0x5eedU);
    DeviceBuffer activation(device(), activation_bits.size() * sizeof(std::uint16_t));
    activation.copy_from_host(activation_bits.data(), activation.size());
    const std::size_t capacity = ops::linear_workspace_capacity_bytes(
        QType::NVFP4, kN, kK, ops::LinearPolicy::AllowA4, kTokens, kTokens);
    DeviceBuffer workspace_storage(device(), std::max<std::size_t>(capacity, 256));
    MemoryArena workspace(workspace_storage.span());
    const Tensor input(activation.span(), DType::BF16, {kK, kTokens});
    return run_staging_race("NVFP4 Linear TMA staging", kLaunch, kTokens,
                            static_cast<std::size_t>(kN) * kTokens,
                            [&](std::int32_t i, MemorySpan output, CudaExecutionView execution) {
                                Tensor destination(output, DType::BF16, {kN, kTokens});
                                ops::linear(input, weights.weight(i), destination,
                                            ops::LinearPolicy::AllowA4, workspace, execution);
                            });
}

int run_swiglu_race() {
    constexpr std::int32_t kGateUp = 34816;
    constexpr std::int32_t kInput  = 5120;
    constexpr std::int32_t kOutput = 17408; // the registered NVFP4 LinearSwiGLU profile
    constexpr std::int32_t kTokens = 256;   // the fused TMA route's floor
    constexpr std::int32_t kLaunch = 6;
    const Weights weights(kLaunch, kGateUp, kInput, 1803U);
    const auto activation_bits = make_activation(kInput, kTokens, 0x5eedU);
    DeviceBuffer activation(device(), activation_bits.size() * sizeof(std::uint16_t));
    activation.copy_from_host(activation_bits.data(), activation.size());
    const std::size_t capacity = ops::linear_swiglu_workspace_capacity_bytes(
        QType::NVFP4, kGateUp, kInput, ops::LinearPolicy::AllowA4, 1, kTokens);
    DeviceBuffer workspace_storage(device(), std::max<std::size_t>(capacity, 256));
    MemoryArena workspace(workspace_storage.span());
    const Tensor input(activation.span(), DType::BF16, {kInput, kTokens});
    return run_staging_race("NVFP4 LinearSwiGLU TMA staging", kLaunch, kTokens,
                            static_cast<std::size_t>(kOutput) * kTokens,
                            [&](std::int32_t i, MemorySpan output, CudaExecutionView execution) {
                                Tensor destination(output, DType::BF16, {kOutput, kTokens});
                                ops::linear_swiglu(input, weights.weight(i), destination,
                                                   ops::LinearPolicy::AllowA4, workspace,
                                                   execution);
                            });
}
} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        int failures = run_linear_race();
        failures += run_swiglu_race();
        std::cout << (failures == 0 ? "OK" : "FAIL")
                  << " NVFP4 TMA staging race (Linear + LinearSwiGLU)\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "NVFP4 TMA staging race: " << error.what() << '\n';
        return 1;
    }
}
