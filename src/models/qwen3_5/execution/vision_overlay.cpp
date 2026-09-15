#include "models/qwen3_5/execution/vision_overlay.h"

#include "models/qwen3_5/execution/vision_overlay.h"

#include "core/device.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/execution/vision.h"
#include "models/qwen3_5/model.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::execution {
namespace overlay_detail {

using OverlayClock = std::chrono::steady_clock;

std::size_t staging_align(std::size_t bytes) {
    constexpr std::size_t kAlign = 256;
    return (bytes + kAlign - 1) / kAlign * kAlign;
}

std::ptrdiff_t byte_delta(const std::byte* target, const std::byte* source) {
    return target - source;
}

Tensor rebase(Tensor tensor, std::ptrdiff_t delta) {
    tensor.data = static_cast<std::byte*>(tensor.data) + delta;
    return tensor;
}

Weight rebase(Weight weight, std::ptrdiff_t delta) {
    const auto shift = [delta](const void* pointer) -> const void* {
        return pointer == nullptr
                   ? nullptr
                   : static_cast<const void*>(static_cast<const std::byte*>(pointer) + delta);
    };
    weight.payload = shift(weight.payload);
    weight.qdata   = shift(weight.qdata);
    weight.qhigh   = shift(weight.qhigh);
    weight.scales  = shift(weight.scales);
    return weight;
}

VisionBlockParameters rebase_layer(const VisionBlockParameters& source, std::ptrdiff_t delta) {
    VisionBlockParameters out = source;
    out.norm1.weight          = rebase(source.norm1.weight, delta);
    out.norm1.bias            = rebase(source.norm1.bias, delta);
    out.norm2.weight          = rebase(source.norm2.weight, delta);
    out.norm2.bias            = rebase(source.norm2.bias, delta);
    out.qkv.weight            = rebase(source.qkv.weight, delta);
    out.qkv_bias              = rebase(source.qkv_bias, delta);
    out.output.weight         = rebase(source.output.weight, delta);
    out.output_bias           = rebase(source.output_bias, delta);
    out.fc1.weight            = rebase(source.fc1.weight, delta);
    out.fc1_bias              = rebase(source.fc1_bias, delta);
    out.fc2.weight            = rebase(source.fc2.weight, delta);
    out.fc2_bias              = rebase(source.fc2_bias, delta);
    return out;
}

} // namespace overlay_detail

VisionWeightStream::VisionWeightStream(DeviceContext& device, const VisionOverlayAssets& assets,
                                       std::byte* staging)
    : device_(device), assets_(assets) {
    using overlay_detail::staging_align;
    const VisionOverlayLayout& layout = assets.layout;
    prelude_ = staging;
    merger_  = prelude_ + staging_align(layout.prelude_bytes);
    slot_[0] = merger_ + staging_align(layout.merger_bytes);
    slot_[1] = slot_[0] + staging_align(layout.slot_bytes);
    for (cudaEvent_t& event : uploaded_) {
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&prelude_event_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&merger_event_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&compute_fence_, cudaEventDisableTiming));

    const std::byte* block = assets.pinned_block;
    CUDA_CHECK(cudaMemcpyAsync(prelude_, block + layout.prelude_begin, layout.prelude_bytes,
                               cudaMemcpyHostToDevice, copy_stream()));
    CUDA_CHECK(cudaEventRecord(prelude_event_, copy_stream()));
    CUDA_CHECK(cudaMemcpyAsync(merger_, block + layout.merger_begin, layout.merger_bytes,
                               cudaMemcpyHostToDevice, copy_stream()));
    CUDA_CHECK(cudaEventRecord(merger_event_, copy_stream()));
    upload_bytes_ = layout.prelude_bytes + layout.merger_bytes;
    reset(device_.stream);
}

VisionWeightStream::~VisionWeightStream() {
    // The window synchronizes the device before restore; events are idle here.
    for (cudaEvent_t event : uploaded_) { (void)cudaEventDestroy(event); }
    (void)cudaEventDestroy(prelude_event_);
    (void)cudaEventDestroy(merger_event_);
    (void)cudaEventDestroy(compute_fence_);
}

VisionParameters VisionWeightStream::window_weights(const VisionParameters& host) const {
    using overlay_detail::byte_delta;
    using overlay_detail::rebase;
    using overlay_detail::rebase_layer;
    const VisionOverlayLayout& layout = assets_.layout;
    const std::byte* block            = assets_.pinned_block;

    VisionParameters out = host;
    const std::ptrdiff_t prelude_delta = byte_delta(prelude_, block + layout.prelude_begin);
    out.patch_embedding.weight = rebase(host.patch_embedding.weight, prelude_delta);
    out.patch_embedding_bias   = rebase(host.patch_embedding_bias, prelude_delta);
    out.position_embedding     = rebase(host.position_embedding, prelude_delta);
    for (std::size_t layer = 0; layer < host.layers.size(); ++layer) {
        const std::ptrdiff_t delta =
            byte_delta(slot_[layer % 2], block + layout.layer_begin[layer]);
        out.layers[layer] = rebase_layer(host.layers[layer], delta);
    }
    const std::ptrdiff_t merger_delta = byte_delta(merger_, block + layout.merger_begin);
    out.merger_norm.weight            = rebase(host.merger_norm.weight, merger_delta);
    out.merger_norm.bias              = rebase(host.merger_norm.bias, merger_delta);
    out.merger_fc1.weight             = rebase(host.merger_fc1.weight, merger_delta);
    out.merger_fc1_bias               = rebase(host.merger_fc1_bias, merger_delta);
    out.merger_fc2.weight             = rebase(host.merger_fc2.weight, merger_delta);
    out.merger_fc2_bias               = rebase(host.merger_fc2_bias, merger_delta);
    return out;
}

void VisionWeightStream::reset(cudaStream_t compute) {
    CUDA_CHECK(cudaEventRecord(compute_fence_, compute));
    CUDA_CHECK(cudaStreamWaitEvent(copy_stream(), compute_fence_, 0));
    next_upload_ = 0;
    upload_next_layer();
    upload_next_layer();
}

void VisionWeightStream::prelude_ready(cudaStream_t compute) {
    CUDA_CHECK(cudaStreamWaitEvent(compute, prelude_event_, 0));
}

void VisionWeightStream::merger_ready(cudaStream_t compute) {
    CUDA_CHECK(cudaStreamWaitEvent(compute, merger_event_, 0));
}

void VisionWeightStream::arrive(std::uint32_t layer, cudaStream_t compute) {
    CUDA_CHECK(cudaStreamWaitEvent(compute, uploaded_[layer % 2], 0));
    const std::uint32_t layers = static_cast<std::uint32_t>(assets_.layout.layer_begin.size());
    if (next_upload_ == layer + 1 && next_upload_ < layers) {
        CUDA_CHECK(cudaEventRecord(compute_fence_, compute));
        CUDA_CHECK(cudaStreamWaitEvent(copy_stream(), compute_fence_, 0));
        upload_next_layer();
    }
}

void VisionWeightStream::upload_next_layer() {
    if (next_upload_ >= static_cast<std::uint32_t>(assets_.layout.layer_begin.size())) { return; }
    const std::uint32_t layer = next_upload_++;
    const VisionOverlayLayout& layout = assets_.layout;
    CUDA_CHECK(cudaMemcpyAsync(slot_[layer % 2],
                               assets_.pinned_block + layout.layer_begin[layer],
                               layout.layer_bytes[layer], cudaMemcpyHostToDevice,
                               copy_stream()));
    CUDA_CHECK(cudaEventRecord(uploaded_[layer % 2], copy_stream()));
    upload_bytes_ += layout.layer_bytes[layer];
}

std::vector<PinnedVisionResult>
encode_items_overlay(DeviceContext& device, const Parameters& parameters,
                     const qwen3_5::PreparedPromptData& prompt, const detail::VisionPrefillPlan& plan,
                     std::size_t first_item, VisionOverlayWindowStats* stats) {
    using overlay_detail::OverlayClock;
    using overlay_detail::staging_align;
    const Model& model = parameters.model;
    if (!model.weights().vision) {
        throw std::logic_error("overlay window requires vision weights");
    }
    if (!model.overlay_vision()) {
        throw std::logic_error("overlay window requires overlay assets");
    }
    const VisionOverlayAssets& assets = *model.overlay_vision();
    EvictableWeightPool& pool         = *assets.pool;
    if (plan.control == nullptr || plan.uses.empty()) {
        throw std::invalid_argument("overlay window has no vision items");
    }
    const std::size_t prepared_begin = plan.control->prepared_item_begin;
    const std::size_t prepared_end   = prepared_begin + plan.control->items.size();
    if (first_item < prepared_begin || first_item > prepared_end ||
        prepared_end > prompt.vision_items.size()) {
        throw std::invalid_argument("overlay window start is outside the prepared items");
    }
    const std::size_t first_control = first_item - prepared_begin;

    const auto& config = model.config().vision.value();
    const auto& params = parameters.vision.value();

    // The window is self-contained: weight staging, encode workspace, and the item output all
    // live in memory borrowed from the evicted weight tail.
    std::size_t max_patches = 0;
    std::uint32_t max_merged = 0;
    for (std::size_t index = first_control; index < plan.control->items.size(); ++index) {
        const qwen3_5::VisionItemControl& control = plan.control->items[index];
        max_patches = std::max(max_patches, control.patch_count);
        max_merged  = std::max(max_merged, static_cast<std::uint32_t>(control.merged_count));
    }
    const auto window_plan =
        VisionContext::plan_overlay_window(config, params, max_patches, max_merged);

    const auto window_start = OverlayClock::now();
    std::size_t mapped      = 0;
    std::byte* staging      =
        pool.evict(staging_align(assets.layout.staging_bytes) + window_plan.capacity_bytes,
                   &mapped);
    std::byte* lease = staging + staging_align(assets.layout.staging_bytes);

    struct RestoreGuard {
        EvictableWeightPool& pool;
        cudaStream_t stream;
        ~RestoreGuard() {
            // On the failure path in-flight work may still reference the overlay range;
            // quiesce before remapping so restore stays safe.
            (void)cudaDeviceSynchronize();
            pool.restore(stream);
        }
    } restore_guard{pool, device.stream};

    const DeviceSpan lease_span{lease, window_plan.capacity_bytes};

    std::vector<PinnedVisionResult> results;
    std::size_t staged_bytes = 0;
    {
        VisionWeightStream stream(device, assets, staging);
        const VisionParameters window_view = stream.window_weights(params);
        const VisionContext context(device, config, window_view);
        results.reserve(prepared_end);
        for (std::size_t skipped = 0; skipped < first_item; ++skipped) {
            results.push_back(PinnedVisionResult{});  // prefix-reused: never consumed
        }
        for (std::size_t index = first_control; index < plan.control->items.size(); ++index) {
            const qwen3_5::VisionItemControl& control = plan.control->items[index];
            const std::size_t prepared_index          = prepared_begin + index;
            const qwen3_5::VisionItem& source         = prompt.vision_items.at(prepared_index);
            const std::size_t patch_elements =
                control.patch_count * static_cast<std::size_t>(config.patch_width());
            const auto& payload = prompt.media_payloads.at(prepared_index);
            if (source.patch_begin != control.patch_begin || payload == nullptr ||
                payload->patch_elements != patch_elements) {
                throw std::invalid_argument("overlay item patch payload has an invalid shape");
            }
            Tensor output = VisionContext::bind_output(lease_span, window_plan,
                                                       control.merged_count);

            if (index != first_control) { stream.reset(device.stream); }
            context.encode(VisionItemView{payload->span(), &control}, output, lease_span,
                           window_plan, &stream);

            const std::size_t embedding_bytes = output.bytes();
            results.push_back(PinnedVisionResult{
                std::make_unique<PinnedHostBuffer>(embedding_bytes), embedding_bytes});
            CUDA_CHECK(cudaMemcpyAsync(results.back().buffer->data(), output.data, embedding_bytes,
                                       cudaMemcpyDeviceToHost, device.stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
        CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
        staged_bytes = stream.uploaded_bytes();
    }

    pool.restore(device.stream);  // the guard's later call becomes a no-op

    if (stats != nullptr) {
        stats->window_seconds  =
            std::chrono::duration<double>(OverlayClock::now() - window_start).count();
        stats->evict_seconds   = pool.last_evict_seconds();
        stats->restore_seconds = pool.last_restore_seconds();
        stats->evicted_bytes   = mapped;
        stats->staged_bytes    = staged_bytes;
    }
    return results;
}

} // namespace ninfer::models::qwen3_5::execution
