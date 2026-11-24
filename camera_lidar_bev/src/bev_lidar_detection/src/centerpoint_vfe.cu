#include "centerpoint_vfe.hpp"

namespace bev_lidar_detection {
namespace {
constexpr int threads = 64;  // one per output channel
constexpr int grid_x = 512;
constexpr int canvas_plane = 512 * 512;
constexpr float voxel = 0.2f, range_min_xy = -51.2f, z_center = -1.0f;  // z_min + voxel_z / 2

// pfn_weights.bin layout: W1[32][11], scale1[32], shift1[32], W2[64][64], scale2[64], shift2[64].
constexpr int w1 = 0, s1 = w1 + 32 * 11, t1 = s1 + 32, w2 = t1 + 32, s2 = w2 + 64 * 64, t2 = s2 + 64;
static_assert(t2 + 64 == centerpoint_pfn_weight_count, "PFN weight layout");

// Exact DynamicPillarVFE: every point of the pillar, max-pooled twice. ReLU outputs are >= 0
// and every pillar has at least one point, so zero-initialised maxima are exact.
__global__ void vfe_kernel(const float4 *__restrict__ points, const float *__restrict__ ages,
                           const int *__restrict__ offsets, const int *__restrict__ flat, int pillars,
                           const float *__restrict__ weights, float *__restrict__ canvas) {
  __shared__ float w[centerpoint_pfn_weight_count];
  __shared__ float mean[3], h1[32], m1[32];
  for (int i = threadIdx.x; i < centerpoint_pfn_weight_count; i += threads) w[i] = weights[i];
  __syncthreads();
  const int t = threadIdx.x;
  for (int p = blockIdx.x; p < pillars; p += gridDim.x) {
    const int begin = offsets[p], end = offsets[p + 1], cell = flat[p];
    const float cx = (cell % grid_x) * voxel + voxel / 2 + range_min_xy;
    const float cy = (cell / grid_x) * voxel + voxel / 2 + range_min_xy;
    if (t < 3) {
      float sum = 0;
      for (int i = begin; i < end; ++i) sum += t == 0 ? points[i].x : t == 1 ? points[i].y : points[i].z;
      mean[t] = sum / static_cast<float>(end - begin);
    }
    __syncthreads();
    const auto layer1 = [&](int i) {
      const float4 q = points[i];
      const float age = ages ? ages[i] : 0.0f;  // sweep time lag (s); 0 for a single sweep
      const float f[11] = {q.x, q.y, q.z, q.w, age, q.x - mean[0], q.y - mean[1], q.z - mean[2],
                           q.x - cx, q.y - cy, q.z - z_center};
      float acc = 0;
#pragma unroll
      for (int k = 0; k < 11; ++k) acc += w[w1 + t * 11 + k] * f[k];
      return fmaxf(acc * w[s1 + t] + w[t1 + t], 0.0f);
    };
    if (t < 32) {
      float m = 0;
      for (int i = begin; i < end; ++i) m = fmaxf(m, layer1(i));
      m1[t] = m;
    }
    __syncthreads();
    float pillar_term = 0;  // W2[:, 32:64] . pillar max, identical for every point
#pragma unroll 8
    for (int k = 0; k < 32; ++k) pillar_term += w[w2 + t * 64 + 32 + k] * m1[k];
    float m2 = 0;
    for (int i = begin; i < end; ++i) {
      if (t < 32) h1[t] = layer1(i);
      __syncthreads();
      float acc = pillar_term;
#pragma unroll 8
      for (int k = 0; k < 32; ++k) acc += w[w2 + t * 64 + k] * h1[k];
      m2 = fmaxf(m2, fmaxf(acc * w[s2 + t] + w[t2 + t], 0.0f));
      __syncthreads();
    }
    canvas[t * canvas_plane + cell] = m2;
  }
}

__global__ void clear_kernel(const int *__restrict__ flat, int pillars, float *__restrict__ canvas) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < pillars * threads) canvas[(i % threads) * canvas_plane + flat[i / threads]] = 0.0f;
}
}  // namespace

cudaError_t launch_centerpoint_vfe(const float4 *points, const float *ages, const int *offsets, const int *flat,
                                   int pillars, const float *weights, float *canvas, cudaStream_t stream) {
  if (pillars <= 0) return cudaSuccess;
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  const int blocks = pillars < sms * 32 ? pillars : sms * 32;
  vfe_kernel<<<blocks, threads, 0, stream>>>(points, ages, offsets, flat, pillars, weights, canvas);
  return cudaGetLastError();
}

cudaError_t launch_centerpoint_clear(const int *flat, int pillars, float *canvas, cudaStream_t stream) {
  if (pillars <= 0) return cudaSuccess;
  const int n = pillars * threads;
  clear_kernel<<<(n + 255) / 256, 256, 0, stream>>>(flat, pillars, canvas);
  return cudaGetLastError();
}
}  // namespace bev_lidar_detection
