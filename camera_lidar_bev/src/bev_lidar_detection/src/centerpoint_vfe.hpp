#pragma once

#include <cuda_runtime_api.h>
#include <vector_types.h>

namespace bev_lidar_detection {

constexpr int centerpoint_pfn_weight_count = 32 * 11 + 32 + 32 + 64 * 64 + 64 + 64;

// Pillar feature network for CenterPoint-PointPillar. points (model frame x, y, z, intensity
// 0..255) are grouped by pillar: pillar p owns points[offsets[p], offsets[p+1]) and sits at
// canvas cell flat[p] = y_idx * 512 + x_idx. ages (same order, seconds behind the newest
// sweep) may be null for a single sweep. Writes 64 features per pillar into the zeroed
// [64][512][512] canvas.
cudaError_t launch_centerpoint_vfe(const float4 *points, const float *ages, const int *offsets, const int *flat,
                                   int pillars, const float *weights, float *canvas, cudaStream_t stream);

// Re-zeroes the canvas cells written for these pillars.
cudaError_t launch_centerpoint_clear(const int *flat, int pillars, float *canvas, cudaStream_t stream);

}  // namespace bev_lidar_detection
