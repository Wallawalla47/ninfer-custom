#include "models/qwen3_5/load/vision_overlay.h"

#include "models/qwen3_5/load/bindings.h"
#include "models/qwen3_5/load/vision_overlay.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5 {
namespace {

struct PinnedRangeIndex {
    std::unordered_map<std::size_t, std::pair<std::size_t, std::size_t>> by_object;

    explicit PinnedRangeIndex(const artifact::MaterializationPlan& plan) {
        by_object.reserve(plan.pinned_objects.size());
        for (const artifact::PinnedPlacement& placement : plan.pinned_objects) {
            by_object.emplace(placement.object.index,
                              std::make_pair(static_cast<std::size_t>(placement.offset),
                                             static_cast<std::size_t>(placement.bytes)));
        }
    }

    // Contiguous extent covering every listed object; throws when the objects are not
    // pinned or leave a gap larger than alignment padding.
    template <class Handles>
    std::pair<std::size_t, std::size_t> extent(const Handles& handles) const {
        std::size_t begin = SIZE_MAX;
        std::size_t end   = 0;
        std::size_t sum   = 0;
        for (const artifact::ObjectHandle handle : handles) {
            const auto it = by_object.find(handle.index);
            if (it == by_object.end()) {
                throw std::logic_error("vision overlay group tensor is not pinned");
            }
            begin = std::min(begin, it->second.first);
            end   = std::max(end, it->second.first + it->second.second);
            sum += it->second.second;
        }
        if (begin == SIZE_MAX || end <= begin || end - begin > sum + handles.size() * 4096) {
            throw std::logic_error("vision overlay group is not contiguous in the pinned block");
        }
        return {begin, end - begin};
    }
};

constexpr std::size_t kStagingAlignment = 256;

std::size_t staging_align(std::size_t bytes) {
    return (bytes + kStagingAlignment - 1) / kStagingAlignment * kStagingAlignment;
}

std::vector<artifact::ObjectHandle>
objects_for(const std::vector<loading::PendingWeight>& pending, WeightId id) {
    std::vector<artifact::ObjectHandle> out;
    for (const auto& part : pending.at(id.index).reference.binding.parts) { out.push_back(part.object); }
    return out;
}

} // namespace

VisionOverlayLayout compute_vision_overlay_layout(const ModelWeights& weights,
                                                  const std::vector<loading::PendingWeight>& pending,
                                                  const artifact::MaterializationPlan& plan) {
    if (!weights.vision) { throw std::logic_error("vision overlay layout requires vision weights"); }
    const VisionWeights& vision = *weights.vision;
    const PinnedRangeIndex index(plan);
    VisionOverlayLayout out;

    {
        const auto handles = [&] {
            std::vector<artifact::ObjectHandle> h;
            for (const WeightId id :
                 {vision.patch_embedding, vision.patch_embedding_bias, vision.position_embedding}) {
                auto part = objects_for(pending, id);
                h.insert(h.end(), part.begin(), part.end());
            }
            return h;
        }();
        auto [begin, bytes] = index.extent(handles);
        out.prelude_begin   = begin;
        out.prelude_bytes   = bytes;
    }

    out.layer_begin.reserve(vision.layers.size());
    out.layer_bytes.reserve(vision.layers.size());
    for (std::size_t layer = 0; layer < vision.layers.size(); ++layer) {
        const VisionBlockWeights& source = vision.layers[layer];
        const auto handles               = [&] {
            std::vector<artifact::ObjectHandle> h;
            for (const WeightId id :
                 {source.norm1.weight, source.norm1.bias, source.norm2.weight, source.norm2.bias,
                  source.query, source.key, source.value, source.query_bias, source.key_bias,
                  source.value_bias, source.output, source.output_bias, source.fc1,
                  source.fc1_bias, source.fc2, source.fc2_bias}) {
                auto part = objects_for(pending, id);
                h.insert(h.end(), part.begin(), part.end());
            }
            return h;
        }();
        auto [begin, bytes] = index.extent(handles);
        out.layer_begin.push_back(begin);
        out.layer_bytes.push_back(bytes);
        out.slot_bytes = std::max(out.slot_bytes, bytes);
    }

    {
        const auto handles = [&] {
            std::vector<artifact::ObjectHandle> h;
            for (const WeightId id : {vision.merger_norm.weight, vision.merger_norm.bias,
                                      vision.merger_fc1, vision.merger_fc1_bias,
                                      vision.merger_fc2, vision.merger_fc2_bias}) {
                auto part = objects_for(pending, id);
                h.insert(h.end(), part.begin(), part.end());
            }
            return h;
        }();
        auto [begin, bytes] = index.extent(handles);
        out.merger_begin    = begin;
        out.merger_bytes    = bytes;
    }

    out.staging_bytes = staging_align(out.prelude_bytes) + staging_align(out.merger_bytes) +
                        2 * staging_align(out.slot_bytes);
    return out;
}

} // namespace ninfer::models::qwen3_5
