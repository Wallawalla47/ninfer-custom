function(ninfer_internal_includes target)
  target_include_directories(${target} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/third_party
    ${PROJECT_SOURCE_DIR}/third_party/utf8proc)
endfunction()

function(ninfer_cuda_archive target)
  set_target_properties(${target} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endfunction()

# Physical device, tensor, weight, layout and graph primitives.
if(WIN32)
  set(NINFER_READ_ONLY_FILE_SOURCE core/read_only_file_win32.cpp)
else()
  set(NINFER_READ_ONLY_FILE_SOURCE core/read_only_file_posix.cpp)
endif()

add_library(ninfer_core STATIC
  ${NINFER_READ_ONLY_FILE_SOURCE}
  core/arena.cu
  core/evictable_weight_pool.cu
  core/device.cu
  core/dtype.cpp
  core/host_worker_pool.cpp
  core/gdn_replay_records.cpp
  core/cyclic_kv_cache.cpp
  core/host_kv_arena.cpp
  core/paged_kv_cache.cpp
  core/linear_attention_state.cpp
  core/layout.cpp
  core/tensor.cpp
  core/weight_view.cpp
  core/decode_graph.cpp)
ninfer_internal_includes(ninfer_core)
ninfer_cuda_archive(ninfer_core)
target_link_libraries(ninfer_core PUBLIC ninfer_cuda_runtime CUDA::nvtx3 CUDA::cuda_driver Threads::Threads)

add_library(ninfer_runtime_support STATIC
  runtime/engine/admission_policy.cpp
  runtime/engine/context_cache/context_cost.cpp
  runtime/engine/context_cache/context_cost_defaults.cpp
  runtime/engine/kv_capacity.cpp
  runtime/engine/public_types.cpp)
ninfer_internal_includes(ninfer_runtime_support)
target_link_libraries(ninfer_runtime_support PUBLIC ninfer_core)

# Generic .ninfer framing, binding, and direct final materialization.
add_library(ninfer_artifact STATIC
  artifact/schema.cpp
  artifact/formats.cpp
  artifact/layouts.cpp
  artifact/file_io.cpp
  artifact/reader.cpp
  artifact/binder.cpp
  artifact/materializer.cpp
  artifact/views.cpp)
ninfer_internal_includes(ninfer_artifact)
target_link_libraries(ninfer_artifact PUBLIC ninfer_core)

# Warp-specialized SM120 NVFP4 kernels use setmaxnreg to transfer registers from producer
# warpgroups to consumer warpgroups. Relocatable device code makes ptxas discard that contract, so
# these self-contained kernels are intentionally compiled in a non-RDC archive and called through
# host launchers from ninfer_ops.
add_library(ninfer_nvfp4_non_rdc STATIC
  ops/linear/nvfp4/nvfp4_w4a4_tma.cu
  ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cu
  ops/softmax_attention/dense/causal_cache/prompt_nvfp4_non_rdc.cu)
ninfer_internal_includes(ninfer_nvfp4_non_rdc)
set_target_properties(ninfer_nvfp4_non_rdc PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
target_compile_options(ninfer_nvfp4_non_rdc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
target_link_libraries(ninfer_nvfp4_non_rdc PRIVATE ninfer_core ninfer_cuda_runtime CUDA::cuda_driver)

# Shared mathematical Ops. The list is intentionally explicit: adding a
# source is a build-boundary decision, not an accidental recursive-glob side effect.
add_library(ninfer_ops STATIC
  ops/weight_input.cpp
  ops/launcher/add_bias.cu
  ops/launcher/argmax.cu
  ops/launcher/cast.cu
  ops/launcher/causal_conv1d.cu
  ops/launcher/embed_gather.cu
  ops/launcher/gdn_gating.cu
  ops/launcher/gelu.cu
  ops/launcher/l2norm.cu
  ops/launcher/layer_norm.cu
  ops/launcher/mtp_pack.cu
  ops/launcher/mtp_round.cu
  ops/launcher/speculative_round.cu
  ops/launcher/position.cu
  ops/launcher/prepare_ragged_prefix.cu
  ops/launcher/prepare_masked_block.cu
  ops/launcher/residual_add.cu
  ops/launcher/rmsnorm.cu
  ops/launcher/rmsnorm_pack_tail.cu
  ops/launcher/rope.cu
  ops/launcher/sampling.cu
  ops/launcher/scalar.cu
  ops/launcher/scatter.cu
  ops/launcher/sigmoid_gate_mul.cu
  ops/launcher/silu_and_mul.cu
  ops/launcher/target_logprobs.cu
  ops/launcher/vision_pos_embed.cu
  ops/rmsnorm_rope/rmsnorm_rope.cpp
  ops/rmsnorm_rope/launch.cu
  ops/context_kv_materialize/context_kv_materialize.cpp
  ops/context_kv_materialize/materialize.cu
  ops/candidate_selector/bf16/candidate_selector_path.cu
  ops/candidate_selector/bf16/candidate_selector_path_plan.cpp
  # Softmax Attention and KV-cache state transitions.
  ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp
  ops/softmax_attention/dense/causal_cache/small_t.cu
  ops/softmax_attention/dense/causal_cache/small_t_fp8.cu
  ops/softmax_attention/dense/causal_cache/small_t_nvfp4.cu
  ops/softmax_attention/dense/causal_cache/small_t_k8v4.cu
  ops/softmax_attention/dense/causal_cache/prompt.cu
  ops/softmax_attention/dense/causal_cache/prompt_fp8.cu
  ops/softmax_attention/dense/causal_cache/prompt_nvfp4.cu
  ops/softmax_attention/dense/causal_cache/prompt_k8v4.cu
  ops/softmax_attention/dense/packed/packed_softmax_attention.cpp
  ops/softmax_attention/dense/packed/launch.cu
  ops/softmax_attention/dense/context/context_softmax_attention.cpp
  ops/softmax_attention/dense/context/launch.cu
  ops/softmax_attention/sliding_window/sliding_window_attention.cpp
  ops/softmax_attention/sliding_window/launch.cu
  ops/kv_cache/append/kv_cache_append.cpp
  ops/kv_cache/append/launch.cu
  ops/kv_cache/append/nvfp4_launch.cu
  ops/kv_cache/append/k8v4_launch.cu
  # Gated DeltaNet Linear Attention core.
  ops/linear_attention/gated_delta_net/gated_delta_net.cpp
  ops/linear_attention/gated_delta_net/replay.cpp
  ops/linear_attention/gated_delta_net/recurrent.cu
  ops/linear_attention/gated_delta_net/chunked/launch.cu
  ops/linear_attention/gated_delta_net/chunked/prepare_wy_wu.cu
  ops/linear_attention/gated_delta_net/chunked/state_passing.cu
  ops/linear_attention/gated_delta_net/chunked/output.cu
  ops/attn_input_proj/bf16/bf16_attn_input_decode.cu
  ops/attn_input_proj/bf16/bf16_attn_input_gemm_mma.cu
  ops/attn_input_proj/bf16/bf16_attn_input_small_t.cu
  ops/attn_input_proj/bf16/bf16_attn_input_plan.cpp
  ops/attn_input_proj/fp8/fp8_attn_input_decode.cu
  ops/attn_input_proj/fp8/fp8_attn_input_small_t.cu
  ops/attn_input_proj/fp8/fp8_attn_input_a16_small_t.cu
  ops/attn_input_proj/fp8/fp8_attn_input_a16_gemm.cu
  ops/attn_input_proj/fp8/fp8_attn_input_a8.cu
  ops/attn_input_proj/fp8/fp8_attn_input_plan.cpp
  ops/attn_input_proj/nvfp4/nvfp4_attn_input_decode.cu
  ops/attn_input_proj/nvfp4/nvfp4_attn_input_small_t.cu
  ops/attn_input_proj/nvfp4/nvfp4_attn_input_w4a4.cu
  ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.cpp
  ops/attn_input_proj/q4_q5/q4_q5_attn_input_gemm_mma.cu
  ops/attn_input_proj/q4_q5/q4_q5_attn_input_small_t.cu
  ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.cpp
  ops/attn_input_proj/q8/q8_attn_input_decode.cu
  ops/attn_input_proj/q8/q8_attn_input_gemm_mma.cu
  ops/attn_input_proj/q8/q8_attn_input_gemm_simt.cu
  ops/attn_input_proj/q8/q8_attn_input_gemm_splitk.cu
  ops/attn_input_proj/q8/q8_dflash2_attn_input.cu
  ops/attn_input_proj/q8/q8_attn_input_plan.cpp
  ops/gdn_input_proj/fp8/fp8_gdn_input_decode.cu
  ops/gdn_input_proj/fp8/fp8_gdn_input_matrix.cu
  ops/gdn_input_proj/fp8/fp8_gdn_input_a8.cu
  ops/gdn_input_proj/fp8/fp8_gdn_input_plan.cpp
  ops/gdn_input_proj/fp8/fp8_gdn_conv_fused.cu
  ops/gdn_input_proj/fp8/fp8_gdn_conv_plan.cpp
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_decode.cu
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_small_t.cu
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_w4a4.cu
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.cpp
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_decode.cu
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_small_t.cu
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_post.cu
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.cpp
  ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_gemm_mma.cu
  ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_independent.cu
  ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_conv_snapshot.cu
  ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.cpp
  ops/gdn_input_proj/gdn_projected_conv.cu
  ops/gdn_input_proj/q8/q8_gdn_input_decode.cu
  ops/gdn_input_proj/q8/q8_gdn_input_gemm_mma.cu
  ops/gdn_input_proj/q8/q8_gdn_input_gemm_splitk.cu
  ops/gdn_input_proj/q8/q8_gdn_input_plan.cpp
  ops/gdn_gating_proj/bf16/bf16_gdn_norm_gating_proj_27.cu
  ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.cu
  ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp
  ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_partial.cu
  ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_reduce.cu
  ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_plan.cpp
  ops/dynamic_grouped_conv/q8/q8_dynamic_grouped_conv_add_materialized.cu
  ops/dynamic_grouped_conv/q8/q8_dynamic_grouped_conv_add_plan.cpp
  ops/linear/bf16/bf16_dispatch.cpp
  ops/linear/bf16/shapes/n14336_k5120.cu
  ops/linear/bf16/shapes/n5120_k6144.cu
  ops/linear/bf16/shapes/n256_k5120.cu
  ops/linear/fp8/fp8_format.cpp
  ops/linear/fp8/fp8_a8.cu
  ops/linear/fp8/fp8_dispatch.cpp
  ops/linear/fp8/shapes/n14336_k5120.cu
  ops/linear/fp8/shapes/n16384_k5120.cu
  ops/linear/fp8/shapes/n34816_k5120.cu
  ops/linear/fp8/shapes/n5120_k6144.cu
  ops/linear/fp8/shapes/n5120_k17408.cu
  ops/linear/fp8/shapes/n248320_k5120.cu
  ops/linear/nvfp4/nvfp4_format.cpp
  ops/linear/nvfp4/nvfp4_w4a4.cu
  ops/linear/nvfp4/nvfp4_dispatch.cpp
  ops/linear/nvfp4/shapes/n14336_k5120.cu
  ops/linear/nvfp4/shapes/n16384_k5120.cu
  ops/linear/nvfp4/shapes/n34816_k5120.cu
  ops/linear/nvfp4/shapes/n5120_k6144.cu
  ops/linear/nvfp4/shapes/n5120_k17408.cu
  ops/linear/q4/q4_rowsplit_gemm_mma.cu
  ops/linear/q4/q4_rowsplit_gemm_simt.cu
  ops/linear/q4/q4_rowsplit_gemv.cu
  ops/linear/q4/q4_dispatch.cpp
  ops/linear/q4/shapes/n1024_k5120.cu
  ops/linear/q4/shapes/n4096_k5120.cpp
  ops/linear/q4/shapes/n5120_k6144.cu
  ops/linear/q4/shapes/n6144_k5120.cu
  ops/linear/q4/shapes/n7168_k5120.cpp
  ops/linear/q4/shapes/n34816_k5120.cpp
  ops/linear/q4/shapes/n131072_k5120.cu
  ops/linear/q4/shapes/n131072_k2048.cu
  ops/linear/q4/shapes/n3456_k1152.cpp
  ops/linear/q4/shapes/n4304_k1152.cpp
  ops/linear/q5/q5_dispatch.cpp
  ops/linear/q5/shapes/n1024_k5120.cpp
  ops/linear/q5/shapes/n6144_k5120.cu
  ops/linear/q5/shapes/n7168_k5120.cu
  ops/linear/q5/shapes/n5120_k6144.cu
  ops/linear/q5/shapes/n5120_k17408.cu
  ops/linear/q5/shapes/n1152_k1152.cpp
  ops/linear/q5/shapes/n1152_k4304.cpp
  ops/linear/q5/q5_rowsplit_gemm_mma.cu
  ops/linear/q5/q5_rowsplit_gemm_simt.cu
  ops/linear/q5/q5_rowsplit_gemv.cu
  ops/linear_add/bf16/bf16_linear_add_decode.cu
  ops/linear_add/bf16/bf16_linear_add_gemm_mma.cu
  ops/linear_add/bf16/bf16_linear_add_small_t.cu
  ops/linear_add/bf16/bf16_linear_add_plan.cpp
  ops/linear_add/nvfp4/nvfp4_linear_add_decode.cu
  ops/linear_add/nvfp4/nvfp4_linear_add_small_t.cu
  ops/linear_add/nvfp4/nvfp4_linear_add_w4a4.cu
  ops/linear_add/nvfp4/nvfp4_linear_add_plan.cpp
  ops/linear_add/fp8/fp8_linear_add_decode.cu
  ops/linear_add/fp8/fp8_linear_add_small_t.cu
  ops/linear_add/fp8/fp8_linear_add_a8.cu
  ops/linear_add/fp8/fp8_linear_add_plan.cpp
  ops/linear_add/q4/q4_linear_add.cu
  ops/linear_add/q5/q5_linear_add_gemm_mma.cu
  ops/linear_add/q5/q5_linear_add_gemm_simt.cu
  ops/linear_add/q5/q5_linear_add_gemv.cu
  ops/linear_add/q5/q5_linear_add_plan.cpp
  ops/linear_add/q8/q8_linear_add_gemm_mma.cu
  ops/linear_add/q8/q8_linear_add_gemm_simt.cu
  ops/linear_add/q8/q8_linear_add_gemm_splitk.cu
  ops/linear_add/q8/q8_linear_add_gemm_capacity.cu
  ops/linear_add/q8/q8_linear_add_gemm_grouped.cu
  ops/linear_add/q8/q8_linear_add_plan.cpp
  ops/linear_swiglu/q4/q4_linear_swiglu_gemm_mma.cu
  ops/linear_swiglu/q4/q4_linear_swiglu_gemv.cu
  ops/linear_swiglu/q4/q4_linear_swiglu_plan.cpp
  ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_decode.cu
  ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_small_t.cu
  ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4.cu
  ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp
  ops/linear_swiglu/fp8/fp8_linear_swiglu_decode.cu
  ops/linear_swiglu/fp8/fp8_linear_swiglu_small_t.cu
  ops/linear_swiglu/fp8/fp8_linear_swiglu_a8.cu
  ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.cpp
  ops/linear_swiglu/q8/q8_linear_swiglu_decode.cu
  ops/linear_swiglu/q8/q8_dflash2_linear_swiglu.cu
  ops/linear_swiglu/q8/q8_linear_swiglu_gemm_mma.cu
  ops/linear_swiglu/q8/q8_linear_swiglu_gemm_splitk.cu
  ops/linear_swiglu/q8/q8_linear_swiglu_plan.cpp
  ops/linear/q6/q6_dispatch.cpp
  ops/linear/q6/shapes/n248320_k5120.cpp
  ops/linear/q6/shapes/n248320_k2048.cpp
  ops/linear/q6/shapes/n1152_k1536.cpp
  ops/linear/q6/q6_rowsplit_gemm_mma.cu
  ops/linear/q6/q6_rowsplit_gemm_simt.cu
  ops/linear/q8/q8_rowsplit_gemm_mma.cu
  ops/linear/q8/q8_rowsplit_gemm_simt.cu
  ops/linear/q8/q8_gemv.cu
  ops/linear/q8/q8_dispatch.cpp
  ops/linear/q8/shapes/n1024_k2048.cpp
  ops/linear/q8/shapes/n1024_k5120.cpp
  ops/linear/q8/shapes/n12288_k2048.cpp
  ops/linear/q8/shapes/n14336_k5120.cu
  ops/linear/q8/shapes/n2048_k16384.cu
  ops/linear/q8/shapes/n2048_k4096.cu
  ops/linear/q8/shapes/n2048_k4608.cpp
  ops/linear/q8/shapes/n248320_k5120.cu
  ops/linear/q8/shapes/n34816_k5120.cu
  ops/linear/q8/shapes/n4608_k4608.cpp
  ops/linear/q8/shapes/n5120_k10240.cu
  ops/linear/q8/shapes/n5120_k17408.cu
  ops/linear/q8/shapes/n5120_k25600.cu
  ops/linear/q8/shapes/n5120_k4608.cpp
  ops/linear/q8/shapes/n5120_k6144.cu
  ops/linear/q8/shapes/n6144_k5120.cu
  ops/linear/q8/shapes/n9216_k2048.cpp
  ops/linear_pair/q8/q8_pair_decode.cu
  ops/linear_pair/q8/q8_pair_gemm_concat.cu
  ops/linear_pair/q8/q8_pair_gemm_mma.cu
  ops/linear_pair/q8/q8_pair_gemm_splitk.cu
  ops/linear_pair/q8/q8_pair_plan.cpp
  ops/linear_topk/linear_topk.cpp
  ops/linear_topk/q8.cu
  ops/linear_topk/q8_m64.cu
  ops/linear_topk/fp8.cu
  ops/linear_topk/fp8_m64.cu
  ops/linear_topk/q4.cu
  ops/linear_topk/q4_m64.cu
  ops/linear_topk/merge.cu
  ops/sparse_moe/decode/sparse_moe_decode_kernels.cu
  ops/sparse_moe/decode/sparse_moe_decode_plan.cpp
  ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu
  ops/sparse_moe/prefill/sparse_moe_prefill_plan.cpp
  ops/sparse_moe/small_t/sparse_moe_small_t_kernels.cu
  ops/sparse_moe/small_t/sparse_moe_small_t_plan.cpp
  ops/linear/linear.cpp
  ops/wrapper/add_bias.cpp
  ops/wrapper/argmax.cpp
  ops/wrapper/cast.cpp
  ops/wrapper/causal_conv1d_silu.cpp
  ops/wrapper/candidate_selector.cpp
  ops/wrapper/dynamic_grouped_conv.cpp
  ops/wrapper/embedding.cpp
  ops/wrapper/gdn_gating.cpp
  ops/wrapper/gdn_gating_proj.cpp
  ops/wrapper/gdn_input_proj.cpp
  ops/wrapper/gelu.cpp
  ops/wrapper/attn_input_proj.cpp
  ops/wrapper/l2norm.cpp
  ops/wrapper/layer_norm.cpp
  ops/wrapper/linear_add.cpp
  ops/wrapper/linear_pair.cpp
  ops/wrapper/linear_swiglu.cpp
  ops/wrapper/mtp_pack.cpp
  ops/wrapper/mtp_round.cpp
  ops/wrapper/speculative_round.cpp
  ops/wrapper/position.cpp
  ops/wrapper/prepare_ragged_prefix.cpp
  ops/wrapper/prepare_masked_block.cpp
  ops/wrapper/residual_add.cpp
  ops/wrapper/rmsnorm.cpp
  ops/wrapper/rmsnorm_pack_tail.cpp
  ops/wrapper/rope.cpp
  ops/wrapper/sampling.cpp
  ops/wrapper/scalar.cpp
  ops/wrapper/scatter.cpp
  ops/wrapper/sigmoid_mul.cpp
  ops/wrapper/silu_mul.cpp
  ops/wrapper/sparse_moe.cpp
  ops/wrapper/target_logprobs.cpp
  ops/wrapper/vision_pos_embed.cpp)
ninfer_internal_includes(ninfer_ops)
ninfer_cuda_archive(ninfer_ops)
target_link_libraries(ninfer_ops PUBLIC ninfer_core PRIVATE ninfer_nvfp4_non_rdc)

# Product-independent text and media mechanisms.
add_library(ninfer_text STATIC
  text/unicode.cpp
  ${PROJECT_SOURCE_DIR}/third_party/utf8proc/utf8proc.c)
ninfer_internal_includes(ninfer_text)
target_compile_definitions(ninfer_text PUBLIC UTF8PROC_STATIC)

# Immutable model data and cold loading, independently buildable from Program and Engine.
add_library(ninfer_model_loading STATIC
  models/registry.cpp
  models/qwen3_5/config.cpp
  models/qwen3_5/model.cpp
  models/qwen3_5/load.cpp
  models/qwen3_5/load/text.cpp
  models/qwen3_5/load/vision.cpp
  models/qwen3_5/load/vision_overlay.cpp
  models/qwen3_5/load/mtp.cpp
  models/qwen3_5/load/dflash.cpp
  models/qwen3_5/load/dflash2.cpp
  models/qwen3_5/load/resources.cpp
  models/qwen3_5/load/prepare.cpp
  models/qwen3_5/frontend/resources.cpp
  models/qwen3_5/frontend/tokenizer.cpp)
ninfer_internal_includes(ninfer_model_loading)
target_link_libraries(ninfer_model_loading PUBLIC ninfer_artifact ninfer_text ninfer_ops)

# The prebuilt Windows vcpkg FFmpeg tree ships without the png decoder, so a
# native PNG path (decode.cpp, gated on NINFER_MEDIA_NATIVE_PNG) decodes PNG
# directly via zlib from the same tree. Cross-platform builds use system FFmpeg
# (which includes png) and skip the native path.
set(_ninfer_media_sources media/decode/decode.cpp)
if(WIN32)
  list(APPEND _ninfer_media_sources media/decode/png_decode.cpp)
endif()
add_library(ninfer_media_decode STATIC
  ${_ninfer_media_sources})
ninfer_internal_includes(ninfer_media_decode)
target_link_libraries(ninfer_media_decode PRIVATE ${NINFER_FFMPEG_TARGET})
if(WIN32)
  add_library(ninfer_zlib INTERFACE)
  set_target_properties(ninfer_zlib PROPERTIES
    INTERFACE_LINK_LIBRARIES "${FFMPEG_VCPKG_TREE}/lib/z.lib")
  target_link_libraries(ninfer_media_decode PRIVATE ninfer_zlib)
  target_compile_definitions(ninfer_media_decode PUBLIC NINFER_MEDIA_NATIVE_PNG)
endif()

add_library(ninfer_model_runtime STATIC
  models/qwen3_5/frontend/frontend.cpp
  models/qwen3_5/frontend/output_session.cpp
  models/qwen3_5/frontend/chat_template.cpp
  models/qwen3_5/frontend/digest.cpp
  models/qwen3_5/frontend/processor.cpp
  models/qwen3_5/frontend/media_cache.cpp
  models/qwen3_5/frontend/tool_call_parser.cpp
  models/qwen3_5/measurement.cpp
  models/qwen3_5/execution/parameters.cpp
  models/qwen3_5/execution/attention.cpp
  models/qwen3_5/execution/ffn.cpp
  models/qwen3_5/execution/gdn.cpp
  models/qwen3_5/execution/mtp.cpp
  models/qwen3_5/execution/text.cpp
  models/qwen3_5/execution/vision.cpp
  models/qwen3_5/execution/vision_overlay.cpp
  models/qwen3_5/execution/draft.cpp
  models/qwen3_5/program/storage/draft_context.cpp
  models/qwen3_5/state/decoder_state.cpp
  models/qwen3_5/state/state_image.cpp
  models/qwen3_5/program/planning/pressure_planner.cpp
  models/qwen3_5/program/planning/startup.cpp
  models/qwen3_5/program/planning/graph_profiles.cpp
  models/qwen3_5/program/program.cpp
  models/qwen3_5/program/program_impl.cpp
  models/qwen3_5/program/context_work.cpp
  models/qwen3_5/program/planning/pressure.cpp
  models/qwen3_5/program/planning/checkpoint_recovery.cpp
  models/qwen3_5/program/transactions/materialization.cpp
  models/qwen3_5/program/storage/context.cpp
  models/qwen3_5/program/transactions/commit.cpp
  models/qwen3_5/program/transactions/capture.cpp
  models/qwen3_5/program/graphs.cpp
  models/qwen3_5/program/decode.cpp
  models/qwen3_5/program/prefill.cpp
  models/qwen3_5/program/planning/request_plan.cpp
  models/qwen3_5/program/speculative/mtp.cpp
  models/qwen3_5/program/speculative/target_verification.cpp
  models/qwen3_5/program/round_buffers.cpp
  models/qwen3_5/program/prefix_identity.cpp
  models/qwen3_5/program/vision_control.cpp
  models/qwen3_5/execution/visual_scatter.cpp)
ninfer_internal_includes(ninfer_model_runtime)
target_link_libraries(ninfer_model_runtime
  PUBLIC ninfer_model_loading ninfer_ops ninfer_runtime_support
  PRIVATE ninfer_media_decode)

if(NINFER_BUILD_MEDIA_ACQUIRE)
  # Product-only path/data/HTTP acquisition. No target package links this library.
  add_library(ninfer_media_acquire STATIC
    product/media_acquire/acquire.cpp)
  ninfer_internal_includes(ninfer_media_acquire)
  target_link_libraries(ninfer_media_acquire PRIVATE ${NINFER_CURL_TARGET})
  if(WIN32)
    target_link_libraries(ninfer_media_acquire PRIVATE ws2_32)
  endif()
endif()

if(NINFER_BUILD_PROMPT_INPUT)
  # Owning command-line message input shared by product entry points and target diagnostics.
  add_library(ninfer_product_prompt_input STATIC
    product/prompt_input/prompt_input.cpp)
  ninfer_internal_includes(ninfer_product_prompt_input)
  target_link_libraries(ninfer_product_prompt_input PRIVATE ninfer_media_acquire)
endif()

if(NINFER_BUILD_APPS OR BUILD_TESTING)
  add_library(ninfer_product_logging STATIC
    product/logging/logging.cpp
    product/logging/pretty_format.cpp
    product/logging/startup_log.cpp)
  ninfer_internal_includes(ninfer_product_logging)
  target_link_libraries(ninfer_product_logging PUBLIC spdlog::spdlog)
endif()

# One cold closed registry plus the opaque public product Engine. Family and exact-target
# directories contribute their sources directly to this single compile owner below.
add_library(ninfer_engine STATIC
  runtime/engine/model_instance.cpp
  runtime/contract/sampling.cpp
  runtime/engine/engine.cpp)
ninfer_internal_includes(ninfer_engine)
target_include_directories(ninfer_engine PUBLIC ${PROJECT_SOURCE_DIR}/include)


target_link_libraries(ninfer_engine
  PRIVATE
    ninfer_model_runtime
    ninfer_artifact
    ninfer_core
    ninfer_runtime_support
    ninfer_ops
    ninfer_text
    ninfer_media_decode)
add_library(ninfer::engine ALIAS ninfer_engine)

if(NINFER_BUILD_SERVE)
  # HTTP protocol/product adapter. No serving header is part of the public API.
  add_library(ninfer_serve STATIC
    serve/http_transport.cpp
    serve/request_validation.cpp
    serve/openai_chat_request.cpp
    serve/openai_chat_response.cpp
    serve/openai_chat_http.cpp
    serve/openai_common.cpp
    serve/openai_responses_request.cpp
    serve/openai_responses_state.cpp
    serve/openai_responses_response.cpp
    serve/anthropic_messages_request.cpp
    serve/anthropic_messages_response.cpp
    serve/anthropic_messages_http.cpp
    serve/openai_responses_store.cpp
    serve/translate.cpp
    serve/generation_service.cpp
    serve/request_events.cpp
    serve/operational_log.cpp
    serve/request_log.cpp
    serve/http_server.cpp
    serve/openai_responses_http.cpp
    serve/serve_options.cpp)
  ninfer_internal_includes(ninfer_serve)
  target_include_directories(ninfer_serve PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
  target_link_libraries(ninfer_serve
    PUBLIC ninfer_engine Threads::Threads
    PRIVATE ninfer_media_acquire ninfer_product_logging ninfer_cuda_runtime)
endif()
