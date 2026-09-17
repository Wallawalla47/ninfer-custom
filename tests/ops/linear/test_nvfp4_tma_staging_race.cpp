// A focused regression test for the Windows-only NVFP4 W4A4 TMA descriptor-staging race.
//
// The Windows staging path copies the over-aligned CUtensorMap descriptor struct from a pinned
// host buffer into a persistent device buffer with cudaMemcpyAsync, which reads the pinned
// source asynchronously on the GPU. With a single pinned source the host can overwrite it before
// the previous async copy has been read, so a launch's kernel then computes against a foreign
// weight. This harness fires the TMA route back-to-back (no inter-launch sync) with distinct
// weights and compares every launch against an isolated (synced) reference: any divergence means
// a launch read a stale or foreign descriptor. It covers both staging sites: the plain Linear
// TMA (launch_tma) and the fused LinearSwiGLU TMA.

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ops/quantized_weight.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string_view>
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::test;

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
// outputs, then fire `launches` back-to-back launches (no inter-launch sync), then bit-exact
// compare. `launch(i, output_device)` enqueues exactly one op call for weight i into
// output_device. The kernels are deterministic for fixed inputs, so a correct run reproduces the
// reference exactly; a staging race feeds a launch a foreign weight and the outputs diverge.
template <typename Launch>
int run_staging_race(std::string_view label, std::int32_t launches, std::int32_t tokens,
                     std::size_t output_elements, Launch launch) {
    std::vector<DeviceBuffer> reference(launches), raced(launches);
    for (std::int32_t i = 0; i < launches; ++i) {
        reference[i] = DeviceBuffer(output_elements * sizeof(std::uint16_t));
        raced[i]     = DeviceBuffer(output_elements * sizeof(std::uint16_t));
    }
    std::vector<std::uint16_t> bits(output_elements);
    cuda_check(cudaStreamSynchronize(nullptr), "idle before reference phase");
    for (std::int32_t i = 0; i < launches; ++i) {
        launch(i, reference[i].p);
        cuda_check(cudaStreamSynchronize(nullptr), "synchronize reference launch");
    }
    // Back-to-back: the host enqueues every staging copy and kernel before the GPU has read the
    // earlier ones, so a single pinned source would be overwritten mid-flight.
    for (std::int32_t i = 0; i < launches; ++i) {
        launch(i, raced[i].p);
    }
    cuda_check(cudaStreamSynchronize(nullptr), "synchronize raced launches");
    long long total_divergent = 0;
    for (std::int32_t i = 0; i < launches; ++i) {
        cuda_check(cudaMemcpy(bits.data(), reference[i].p, output_elements * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost),
                   "copy reference output");
        const std::vector<std::uint16_t> reference_bits(bits.begin(), bits.end());
        cuda_check(cudaMemcpy(bits.data(), raced[i].p, output_elements * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost),
                   "copy raced output");
        long long divergent = 0;
        for (std::size_t j = 0; j < output_elements; ++j) {
            if (bits[j] != reference_bits[j]) { ++divergent; }
        }
        if (divergent != 0) {
            std::cerr << label << " T=" << tokens << " launch=" << i << ": " << divergent << '/'
                      << output_elements << " output elements diverged from the isolated "
                         "reference\n";
        }
        total_divergent += divergent;
    }
    return total_divergent == 0 ? 0 : 1;
}

int run_linear_race() {
    constexpr std::int32_t kN      = 5120;
    constexpr std::int32_t kK      = 6144;  // N5120K6144: the smallest registered TMA problem
    constexpr std::int32_t kTokens = 1024;  // this shape takes the TMA route from T=1024
    constexpr std::int32_t kLaunch = 8;
    std::vector<quantized_weight::PackedWeight> host_weights(kLaunch);
    std::vector<DeviceBuffer> device_weights(kLaunch);
    for (std::int32_t i = 0; i < kLaunch; ++i) {
        host_weights[i] = make_nvfp4(kN, kK, 723U + static_cast<std::uint32_t>(i));
        device_weights[i] = DeviceBuffer(host_weights[i].payload.size());
        device_weights[i].copy_from_host(host_weights[i].payload.data(), device_weights[i].bytes);
    }
    const std::vector<std::uint16_t> activation_bits = make_activation(kK, kTokens, 0x5eedU);
    DeviceBuffer device_activation(activation_bits.size() * sizeof(std::uint16_t));
    device_activation.copy_from_host(activation_bits.data(), device_activation.bytes);
    const std::size_t capacity = ops::linear_workspace_capacity_bytes(
        QType::NVFP4, kN, kK, ops::LinearPolicy::AllowA4, kTokens, kTokens);
    DeviceArena workspace(std::max<std::size_t>(capacity, 256));
    Tensor input(device_activation.p, DType::BF16, {kK, kTokens});
    return run_staging_race(
        "NVFP4 Linear TMA staging", kLaunch, kTokens,
        static_cast<std::size_t>(kN) * kTokens,
        [&](std::int32_t i, void* output) {
            const Weight weight = host_weights[i].device_weight(device_weights[i].p);
            Tensor destination(output, DType::BF16, {kN, kTokens});
            ops::linear(input, weight, destination, ops::LinearPolicy::AllowA4, workspace,
                        nullptr);
        });
}

int run_swiglu_race() {
    constexpr std::int32_t kGateUp = 34816;
    constexpr std::int32_t kInput  = 5120;
    constexpr std::int32_t kOutput = 17408;  // the registered NVFP4 LinearSwiGLU profile
    constexpr std::int32_t kTokens = 256;    // the fused TMA route's floor
    constexpr std::int32_t kLaunch = 6;
    std::vector<quantized_weight::PackedWeight> host_weights(kLaunch);
    std::vector<DeviceBuffer> device_weights(kLaunch);
    for (std::int32_t i = 0; i < kLaunch; ++i) {
        host_weights[i] = make_nvfp4(kGateUp, kInput, 1803U + static_cast<std::uint32_t>(i));
        device_weights[i] = DeviceBuffer(host_weights[i].payload.size());
        device_weights[i].copy_from_host(host_weights[i].payload.data(), device_weights[i].bytes);
    }
    const std::vector<std::uint16_t> activation_bits = make_activation(kInput, kTokens, 0x5eedU);
    DeviceBuffer device_activation(activation_bits.size() * sizeof(std::uint16_t));
    device_activation.copy_from_host(activation_bits.data(), device_activation.bytes);
    const std::size_t capacity = ops::linear_swiglu_workspace_capacity_bytes(
        QType::NVFP4, kGateUp, kInput, ops::LinearPolicy::AllowA4, 1, kTokens);
    DeviceArena workspace(std::max<std::size_t>(capacity, 256));
    Tensor input(device_activation.p, DType::BF16, {kInput, kTokens});
    return run_staging_race(
        "NVFP4 LinearSwiGLU TMA staging", kLaunch, kTokens,
        static_cast<std::size_t>(kOutput) * kTokens,
        [&](std::int32_t i, void* output) {
            const Weight weight = host_weights[i].device_weight(device_weights[i].p);
            Tensor destination(output, DType::BF16, {kOutput, kTokens});
            ops::linear_swiglu(input, weight, destination, ops::LinearPolicy::AllowA4, workspace,
                               nullptr);
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
