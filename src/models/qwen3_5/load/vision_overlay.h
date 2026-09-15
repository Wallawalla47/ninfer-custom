#pragma once

#include "artifact/materializer.h"
#include "core/evictable_weight_pool.h"
#include "models/qwen3_5/weights.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer::models::qwen3_5 {

namespace loading {
struct PendingWeight;
}

// Byte ranges of the vision groups inside the pinned weight block, used by the
// overlay window to stage weights through borrowed device memory. Ranges are
// contiguous by binding order; slot_bytes is the largest single layer range.
struct VisionOverlayLayout {
    std::size_t prelude_begin = 0;   // patch embedding .. position embedding
    std::size_t prelude_bytes = 0;
    std::vector<std::size_t> layer_begin;
    std::vector<std::size_t> layer_bytes;
    std::size_t merger_begin  = 0;   // merger fc1 .. merger norm bias
    std::size_t merger_bytes  = 0;
    std::size_t slot_bytes    = 0;
    std::size_t staging_bytes = 0;   // prelude + merger + two layer slots, aligned
};

// Runtime assets the overlay window needs, published on the model when the engine runs
// with VisionResidency::Overlay. The pool pointer is non-const because the window calls
// the pool's evict()/restore() during an overlay encode.
struct VisionOverlayAssets {
    EvictableWeightPool* pool               = nullptr;
    const std::byte* pinned_block           = nullptr;
    std::size_t pinned_bytes                = 0;
    std::size_t ladder_bytes                = 0;  // evictable tail a window may borrow
    VisionOverlayLayout layout;
};

// Resolve each vision group's contiguous byte range in the pinned block. Throws when a
// group's objects are not all pinned or are not packed contiguously (gap beyond alignment).
[[nodiscard]] VisionOverlayLayout compute_vision_overlay_layout(
    const ModelWeights& weights, const std::vector<loading::PendingWeight>& pending,
    const artifact::MaterializationPlan& plan);

} // namespace ninfer::models::qwen3_5
