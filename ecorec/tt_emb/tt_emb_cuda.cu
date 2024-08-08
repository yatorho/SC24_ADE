
#include <ATen/ATen.h>
#include <assert.h>

#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
// #include <ATen/CUDAGeneratorImpl.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <stdio.h>
#include <iostream>
#include "ATen/ops/_unique.h"
#include "ATen/ops/unique_consecutive.h"
#include "c10/core/TensorOptions.h"
#include "tt_cuda_utils.cuh"

#define WARP_SIZE 32
#define eps 1e-5
#define MAX_BATCH_SIZE 8192

#define checkKernelErrors(expr)                                                         \
  do {                                                                                  \
    expr;                                                                               \
                                                                                        \
    cudaError_t __err = cudaGetLastError();                                             \
    if (__err != cudaSuccess) {                                                         \
      printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, cudaGetErrorString(__err)); \
      abort();                                                                          \
    }                                                                                   \
  } while (0)

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)
#define CHECK_INPUT_ARRAY(x) \
  for (auto t : x) {         \
    CHECK_INPUT(t);          \
  }

using namespace at;

enum {
  OPTIM_SGD = 0,
  OPTIM_ADAGRAD = 1,
  OPTIM_DENSE = 2,
};

inline void cuda_gemm_batched_fp32_fp32(
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    float* alpha,
    void** a_array,
    int lda,
    void** b_array,
    int ldb,
    float* beta,
    void** c_array,
    int ldc,
    int batch_count) {
  if (batch_count <= 0) {
    return;
  }

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasSetStream(handle, c10::cuda::getCurrentCUDAStream());

  cublasGemmBatchedEx(
      handle,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      a_array,
      CUDA_R_32F,
      lda,
      b_array,
      CUDA_R_32F,
      ldb,
      beta,
      c_array,
      CUDA_R_32F,
      ldc,
      batch_count,
      CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT);
}

inline void stream_cuda_gemm_batched_fp32_fp32(
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    float* alpha,
    void** a_array,
    int lda,
    void** b_array,
    int ldb,
    float* beta,
    void** c_array,
    int ldc,
    int batch_count,
    cudaStream_t stream) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasSetStream(handle, stream);
  cublasGemmBatchedEx(
      handle,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      a_array,
      CUDA_R_32F,
      lda,
      b_array,
      CUDA_R_32F,
      ldb,
      beta,
      c_array,
      CUDA_R_32F,
      ldc,
      batch_count,
      CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT);
}

__global__ void reduce_output_kernel(
    int32_t N, // batch cnt
    int32_t D, // feature dim
    int32_t tr_dim, // tr_output_dim
    const int64_t* __restrict__ rowidx,
    const int64_t* __restrict__ unqi_keys,
    const float* __restrict__ uni_output,
    float* __restrict__ reduce_output) {
  int32_t indice_id = blockIdx.x * blockDim.y + threadIdx.y;
  if (indice_id >= N) {
    // don't have *warp* divergence since we launch full warps in blockDim.x,
    // so we can just exit this warp entirely.
    return;
  }
  // check if this warp is responsible for this whole segment.
  bool segment_start = (indice_id == 0 || rowidx[indice_id - 1] != rowidx[indice_id]);
  if (!segment_start) {
    // don't have *warp* divergence since we launch full warps in blockDim.x,
    // so we can just exit this warp entirely.
    return;
  }
  int64_t row_index = rowidx[indice_id];
  // now, find the end of the segment (and thus the segment length `SL`).
  int32_t SL = 1;
  while (indice_id + SL < N && rowidx[indice_id + SL] == row_index) {
    SL += 1;
  }
  for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
    Vec4T<float> sum(&reduce_output[row_index * D + d * 4]);
    for (int32_t sl = 0; sl < SL; ++sl) {
      int64_t idx = __ldg(&unqi_keys[indice_id + sl]);
      Vec4T<float> tr(&uni_output[idx * tr_dim + d * 4]);
      sum.acc.x += tr.acc.x;
      sum.acc.y += tr.acc.y;
      sum.acc.z += tr.acc.z;
      sum.acc.w += tr.acc.w;
    }
    sum.store(&reduce_output[row_index * D + d * 4]);
  }
}

__global__ void batched_reduce_output_kernel(
    int32_t N, // batch cnt
    int32_t D, // feature dim
    int32_t tr_dim, // tr_output_dim
    const int64_t* __restrict__ rowidx,
    const int64_t* __restrict__ back_rowidx,
    const float* __restrict__ uni_output,
    float* __restrict__ reduce_output) {
  int32_t n = blockIdx.x * blockDim.y + threadIdx.y;
  if (n >= N) {
    return;
  }

  auto store_idx = __ldg(&rowidx[n]);
  auto load_idx = __ldg(&back_rowidx[n]);
  for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
    float4 tr = *(float4*)&uni_output[load_idx * tr_dim + d * 4];

    atomicAdd(&reduce_output[store_idx * D + d * 4], tr.x);
    atomicAdd(&reduce_output[store_idx * D + d * 4 + 1], tr.y);
    atomicAdd(&reduce_output[store_idx * D + d * 4 + 2], tr.z);
    atomicAdd(&reduce_output[store_idx * D + d * 4 + 3], tr.w);
  }
}

__global__ void build_reusing_matrix(
    int32_t N,
    int32_t mat_cols,
    const int64_t* __restrict__ tt_idxa,
    const int64_t* __restrict__ tt_idxb,
    int32_t* __restrict__ mat_ptr) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;

  if (n >= N) {
    return;
  }

  int64_t idxa = __ldg(&tt_idxa[n]);
  int64_t idxb = __ldg(&tt_idxb[n]);
  int64_t idx = idxa * mat_cols + idxb;
  atomicAdd(&mat_ptr[idx], 1);
}

std::vector<at::Tensor> BuildReusingMatrices(
    // at::Tensor indices,
    const std::vector<at::Tensor>& tt_indices,
    const std::vector<int>& tt_p_shapes) {
  std::vector<at::Tensor> reusing_mats;
  auto num_indices = tt_indices[0].numel();

  int32_t threads = 256;
  int32_t blocks = (num_indices + threads - 1) / threads;

  for (size_t i = 0; i < tt_p_shapes.size() - 1; ++i) {
    at::Tensor reusing_mat =
        at::zeros({tt_p_shapes[i], tt_p_shapes[i + 1]}, tt_indices[i].options().dtype(at::kInt));

    build_reusing_matrix<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        num_indices,
        tt_p_shapes[i + 1],
        tt_indices[i].data_ptr<int64_t>(),
        tt_indices[i + 1].data_ptr<int64_t>(),
        reusing_mat.data_ptr<int32_t>());

    reusing_mats.push_back(reusing_mat);
  }

  return reusing_mats;
}

__global__ void bce_prepare_part1_batch_gemm_pointers(
    int32_t num_uni,
    int32_t front_num,
    int32_t front_gemm_cnt,
    int32_t front_cache_dim,
    int32_t back_cache_dim,

    const int64_t* __restrict__ bce_idx,
    const int64_t* __restrict__ bce_map,
    const int64_t* __restrict__ tt_p_shape,
    const int64_t* __restrict__ tt_q_shape,
    const int64_t* __restrict__ tt_ranks,
    float* __restrict__ tt_core_0,
    float* __restrict__ tt_core_1,
    float* __restrict__ tt_core_2,
    float* __restrict__ front_trs,
    float* __restrict__ back_trs,
    float** __restrict__ a_ptr,
    float** __restrict__ b_ptr,
    float** __restrict__ c_ptr,
    int32_t* __restrict__ front_flag,
    int32_t* __restrict__ back_flag,
    int32_t* __restrict__ front_group_idx,
    int32_t* __restrict__ back_group_idx,
    float** __restrict__ front_group_map,
    float** __restrict__ back_group_map) {
  int32_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= num_uni) {
    return;
  }

  int32_t idx_start = n;
  int32_t idx_end = n + 1;

  for (int32_t i = idx_start; i < idx_end; ++i) {
    // int64_t bce_key = __ldg(&bce_map[i]);
    int64_t idx = __ldg(&bce_idx[n]);

    int32_t group1 = idx / tt_p_shape[2]; // group1 = I1 * tt_p_shape[1] + I2
    int I3 = idx % tt_p_shape[2];
    int I2 = group1 % tt_p_shape[1];
    // int I1 = group1 / tt_p_shape[1];
    int I1 = idx / (tt_p_shape[1] * tt_p_shape[2]);

    int32_t group2 = I2 * tt_p_shape[2] + I3; // group2 = I2 * tt_p_shape[2] + I3


    if (n < front_num) {
      if (atomicCAS(front_flag + group1, 0, 1) == 0) {
        int cache_idx = atomicAdd(front_group_idx, 1);
        // printf("Part1 F: n: %d, bce_key: %ld, idx: %ld, cache_idx: %d\n", n,
        // idx, idx, cache_idx);

        a_ptr[cache_idx] = tt_core_1 + I2 * (tt_ranks[1] * tt_q_shape[1] * tt_ranks[2]);
        b_ptr[cache_idx] = tt_core_0 + I1 * (tt_q_shape[0] * tt_ranks[1]);
        c_ptr[cache_idx] = front_trs + cache_idx * front_cache_dim;
        front_group_map[group1] = front_trs + cache_idx * front_cache_dim;
      }
    } else {
      if (atomicCAS(back_flag + group2, 0, 1) == 0) {
        int cache_idx = atomicAdd(back_group_idx, 1);
        // printf("Part1 B: n: %d, bce_key: %ld, idx: %ld, cache_idx: %d\n", n,
        // idx, idx, cache_idx);

        a_ptr[front_gemm_cnt + cache_idx] = tt_core_2 + I3 * (tt_q_shape[2] * tt_ranks[2]);
        b_ptr[front_gemm_cnt + cache_idx] =
            tt_core_1 + I2 * (tt_ranks[1] * tt_q_shape[1] * tt_ranks[2]);
        c_ptr[front_gemm_cnt + cache_idx] = back_trs + cache_idx * back_cache_dim;
        back_group_map[group2] = back_trs + cache_idx * back_cache_dim;
      }
    }
  }
}

__global__ void prepare_3cores_uidx_batched_batch_part1_gemm_pointers(
    int32_t num_uni,
    int32_t cache_dim,

    const int64_t* __restrict__ indices,
    const int64_t* __restrict__ tt_p_shape,
    const int64_t* __restrict__ tt_q_shape,
    const int64_t* __restrict__ tt_ranks,
    float* __restrict__ tt_core_0,
    float* __restrict__ tt_core_1,
    float* __restrict__ trs,
    float** __restrict__ a_ptr,
    float** __restrict__ b_ptr,
    float** __restrict__ c_ptr,
    int32_t* __restrict__ group_flag,
    int32_t* __restrict__ group_idx,
    float** __restrict__ group_map) {
  int32_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= num_uni) {
    return;
  }

  int32_t idx_start = n;
  int32_t idx_end = n + 1;

  for (int32_t i = idx_start; i < idx_end; ++i) {
    int64_t idx = __ldg(&indices[n]);

    int32_t group1 = idx / tt_p_shape[2]; // group1 = I1 * tt_p_shape[1] + I2

    if (atomicCAS(group_flag + group1, 0, 1) == 0) {
      int cache_idx = atomicAdd(group_idx, 1);
      int I2 = group1 % tt_p_shape[1];
      // int I1 = group1 / tt_p_shape[1];
      int I1 = idx / (tt_p_shape[1] * tt_p_shape[2]);

      a_ptr[cache_idx] = tt_core_1 + I2 * (tt_ranks[1] * tt_q_shape[1] * tt_ranks[2]);
      b_ptr[cache_idx] = tt_core_0 + I1 * (tt_q_shape[0] * tt_ranks[1]);
      c_ptr[cache_idx] = trs + cache_idx * cache_dim;
      group_map[group1] = trs + cache_idx * cache_dim;
    }
  }
}

__global__ void prepare_3cores_uidx_batched_batch_part2_gemm_pointers(
    int32_t num_uni,
    int32_t part1_gemm_cnt,
    int32_t output_dim, // q1*q2*q3

    const int64_t* __restrict__ indices,
    float* __restrict__ uni_output,
    const int64_t* __restrict__ tt_p_shape,
    const int64_t* __restrict__ tt_q_shape,
    const int64_t* __restrict__ tt_ranks,
    float* __restrict__ tt_core_2,
    float** __restrict__ a_ptr,
    float** __restrict__ b_ptr,
    float** __restrict__ c_ptr,
    float** __restrict__ group_map) {
  int32_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= num_uni) {
    return;
  }

  int32_t idx_start = n;
  int32_t idx_end = n + 1;

  for (int32_t i = idx_start; i < idx_end; ++i) {
    int64_t idx = __ldg(&indices[n]);

    int32_t group1 = idx / tt_p_shape[2]; // group1 = I1 * tt_p_shape[1] + I2
    int I3 = idx % tt_p_shape[2];
    int I2 = group1 % tt_p_shape[1];
    // int I1 = group1 / tt_p_shape[1];
    int I1 = idx / (tt_p_shape[1] * tt_p_shape[2]);

    a_ptr[part1_gemm_cnt + n] = tt_core_2 + I3 * (tt_q_shape[2] * tt_ranks[2]);
    b_ptr[part1_gemm_cnt + n] = group_map[group1]; // from cache
    c_ptr[part1_gemm_cnt + n] = uni_output + n * output_dim;
  }
}

__global__ void bce_prepare_part2_batch_gemm_pointers(
    int32_t num_uni,
    int32_t front_num,
    // int32_t feature_dim,  // TODO: when feature dim is not equal with q1*q2*q3
    int32_t output_dim, // q1*q2*q3
    int32_t bce_part1_gemm_cnt,

    const int64_t* __restrict__ bce_idx,
    const int64_t* __restrict__ bce_map,
    const int64_t* __restrict__ tt_p_shape,
    const int64_t* __restrict__ tt_q_shape,
    const int64_t* __restrict__ tt_ranks,
    float* __restrict__ tt_core_0,
    float* __restrict__ tt_core_2,
    float* __restrict__ uni_output,
    float** __restrict__ a_ptr,
    float** __restrict__ b_ptr,
    float** __restrict__ c_ptr,
    float** __restrict__ front_group_map,
    float** __restrict__ back_group_map) {
  int32_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= num_uni) {
    return;
  }

  int32_t idx_start = n;
  int32_t idx_end = n + 1;

  for (int32_t i = idx_start; i < idx_end; ++i) {
    int64_t idx = __ldg(&bce_idx[n]);

    int32_t group1 = idx / tt_p_shape[2];
    int I3 = idx % tt_p_shape[2];
    int I2 = group1 % tt_p_shape[1];
    int I1 = idx / (tt_p_shape[1] * tt_p_shape[2]);

    int32_t group2 = I2 * tt_p_shape[2] + I3;

    if (n < front_num) {
      a_ptr[bce_part1_gemm_cnt + n] = tt_core_2 + I3 * (tt_q_shape[2] * tt_ranks[2]);
      b_ptr[bce_part1_gemm_cnt + n] = front_group_map[group1]; // from cache
      c_ptr[bce_part1_gemm_cnt + n] = uni_output + bce_map[n] * output_dim;
    } else {
      a_ptr[bce_part1_gemm_cnt + n] = back_group_map[group2];
      b_ptr[bce_part1_gemm_cnt + n] = tt_core_0 + I1 * (tt_q_shape[0] * tt_ranks[1]);
      c_ptr[bce_part1_gemm_cnt + n] = uni_output + bce_map[n] * output_dim;
    }
  }
}

__global__ void bce_prepare_batch_gemm_pointers(
    int32_t num_uni,
    int32_t front_num,
    int32_t num_embedding, // for assert
    int32_t feature_dim, // TODO: when feature dim is not equal with q1*q2*q3
    int32_t output_length, // q1*q2*q3
    int32_t front_gemm_cnt,
    int32_t bce_part1_gemm_cnt,
    int32_t front_cache_dim,
    int32_t back_cache_dim,

    const int64_t* __restrict__ bce_idx,
    const int64_t* __restrict__ bce_map,
    const int64_t* __restrict__ tt_p_shape,
    const int64_t* __restrict__ tt_q_shape,
    const int64_t* __restrict__ tt_ranks,
    float* __restrict__ tt_core_0,
    float* __restrict__ tt_core_1,
    float* __restrict__ tt_core_2,
    float* __restrict__ front_trs,
    float* __restrict__ back_trs,
    float* __restrict__ uni_output,
    float** __restrict__ a_ptr,
    float** __restrict__ b_ptr,
    float** __restrict__ c_ptr,
    int32_t* __restrict__ front_flag,
    int32_t* __restrict__ back_flag,
    int32_t* __restrict__ front_group_idx,
    int32_t* __restrict__ back_group_idx,
    float** __restrict__ front_group_map,
    float** __restrict__ back_group_map) {
  int32_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= num_uni) {
    return;
  }

  int32_t idx_start = n;
  int32_t idx_end = n + 1;

  for (int32_t i = idx_start; i < idx_end; ++i) {
    int64_t idx = __ldg(&bce_map[i]);
    idx = __ldg(&bce_idx[idx]);

    int32_t group1 = idx / tt_p_shape[2]; // group1 = I1 * tt_p_shape[1] + I2
    int I3 = idx % tt_p_shape[2];
    int I2 = group1 % tt_p_shape[1];
    // int I1 = group1 / tt_p_shape[1];
    int I1 = idx / (tt_p_shape[1] * tt_p_shape[2]);

    int32_t group2 = I2 * tt_p_shape[2] + I3; // group2 = I2 * tt_p_shape[2] + I3

    if (n < front_num) {
      if (atomicCAS(front_flag + group1, 0, 1) == 0) {
        int cache_idx = atomicAdd(front_group_idx, 1);

        a_ptr[cache_idx] = tt_core_1 + I2 * (tt_ranks[1] * tt_q_shape[1] * tt_ranks[2]);
        b_ptr[cache_idx] = tt_core_0 + I1 * (tt_q_shape[0] * tt_ranks[1]);
        c_ptr[cache_idx] = front_trs + cache_idx * front_cache_dim;
        front_group_map[group1] = front_trs + cache_idx * front_cache_dim;
        // printf("n: %d, group: %d, front_group_map[group1] = %p\n", n, group1,
        // front_group_map[group1]);
        front_flag[group1] = 0;
      }


      while (atomicCAS(front_flag + group1, 1, 0) != 0) {
      }
      a_ptr[bce_part1_gemm_cnt + n] = tt_core_2 + I3 * (tt_q_shape[2] * tt_ranks[2]);
      b_ptr[bce_part1_gemm_cnt + n] = front_group_map[group1]; // from cache
      c_ptr[bce_part1_gemm_cnt + n] = uni_output + n * feature_dim;
    } else {
      printf("FATAL ERROR!!! THIS SHOULD NEVER BE OCCURED!\n");
      if (atomicCAS(back_flag + group2, 0, 1) == 0) {
        int cache_idx = atomicAdd(back_group_idx, 1);

        a_ptr[front_gemm_cnt + cache_idx] = tt_core_2 + I3 * (tt_q_shape[2] * tt_ranks[2]);
        b_ptr[front_gemm_cnt + cache_idx] =
            tt_core_1 + I2 * (tt_ranks[1] * tt_q_shape[1] * tt_ranks[2]);
        c_ptr[front_gemm_cnt + cache_idx] = back_trs + cache_idx * back_cache_dim;
        back_group_map[group2] = back_trs + cache_idx * back_cache_dim;

        back_flag[group2] = 0;
      }

      while (atomicCAS(back_flag + group2, 1, 0) != 0) {
      }

      a_ptr[bce_part1_gemm_cnt + n] = back_group_map[group2];
      b_ptr[bce_part1_gemm_cnt + n] = tt_core_0 + I1 * (tt_q_shape[0] * tt_ranks[1]);
      c_ptr[bce_part1_gemm_cnt + n] = uni_output + n * feature_dim;
    }
  }
}

__global__ void compute_BCE_batch_gemm_counts_kernel(
    int32_t front_num,
    int32_t N,
    const int64_t* __restrict__ bce_idx,
    const int64_t* __restrict__ tt_p_shape,
    int32_t* __restrict__ front_group_flags,
    int32_t* __restrict__ back_group_flags,
    int32_t* __restrict__ front_gemm_count,
    int32_t* __restrict__ back_gemm_count) {
  int32_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) {
    return;
  }

  int32_t idx_start = n;
  int32_t idx_end = n + 1;

  for (int32_t i = idx_start; i < idx_end; ++i) {
    int64_t idx = __ldg(&bce_idx[i]);
    int32_t group1 = idx / tt_p_shape[2]; // group1 = I1 * tt_p_shape[1] + I2
    if (n < front_num) {
      if (atomicCAS(front_group_flags + group1, 0, 1) == 0) {
        atomicAdd(front_gemm_count, 1);
      }
    } else {
      int I3 = idx % tt_p_shape[2];
      int I2 = group1 % tt_p_shape[1];
      int32_t group2 = I2 * tt_p_shape[2] + I3; // group2 = I2 * tt_p_shape[2] + I3

      if (atomicCAS(back_group_flags + group2, 0, 1) == 0) {
        atomicAdd(back_gemm_count, 1);
      }
    }
  }
}

void compute_BCE_batch_gemm_counts(
    int32_t bce_front_num,
    int32_t num_uni,
    at::Tensor bce_idx,
    at::Tensor tensor_p_shapes,

    at::Tensor front_group_flags,
    at::Tensor back_group_flags,
    at::Tensor front_gemm_count_t,
    at::Tensor back_gemm_count_t) {
  int32_t threads = (num_uni > 256) ? 256 : 32;
  int32_t blocks = (num_uni + threads - 1) / threads;

  compute_BCE_batch_gemm_counts_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      bce_front_num,
      num_uni,
      bce_idx.data_ptr<int64_t>(),
      (const int64_t*)tensor_p_shapes.data_ptr(),
      front_group_flags.data_ptr<int32_t>(),
      back_group_flags.data_ptr<int32_t>(),
      front_gemm_count_t.data_ptr<int32_t>(),
      back_gemm_count_t.data_ptr<int32_t>());
}

__global__ void compute_gemm_counts_kernel(
    int32_t N,
    const int64_t* __restrict__ indices,
    const int64_t* __restrict__ tt_p_shape,
    int32_t* __restrict__ group_flags,
    int32_t* __restrict__ gemm_count) {
  int32_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) {
    return;
  }

  int32_t idx_start = n;
  int32_t idx_end = n + 1;

  for (int32_t i = idx_start; i < idx_end; ++i) {
    int64_t idx = __ldg(&indices[i]);
    int32_t group = idx / tt_p_shape[2]; // group1 = I1 * tt_p_shape[1] + I2
    if (atomicCAS(group_flags + group, 0, 1) == 0) {
      atomicAdd(gemm_count, 1);
    }
  }
}

void compute_gemm_counts(
    at::Tensor indices,
    at::Tensor tensor_p_shapes,

    at::Tensor group_flags,
    at::Tensor gemm_count_t) {
  int32_t num_indices = indices.numel();
  int32_t threads = (num_indices > 256) ? 256 : 32;
  int32_t blocks = (num_indices + threads - 1) / threads;

  compute_gemm_counts_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      num_indices,
      indices.data_ptr<int64_t>(),
      (const int64_t*)tensor_p_shapes.data_ptr(),
      group_flags.data_ptr<int32_t>(),
      gemm_count_t.data_ptr<int32_t>());
}

int32_t compute_uidx_batch_part1_gemm_cnt(
    const std::vector<int>& tt_p_shapes, //[i1,i2,i3]
    const std::vector<int>& tt_q_shapes, //[j1,j2,j3]
    const std::vector<int>& tt_ranks, //[1,r1,r2,1]
    Tensor tensor_p_shapes, //[i1,i2,i3]

    at::Tensor unique_values,
    const std::vector<Tensor>& tt_cores,
    at::Tensor gemm_count_t,
    at::Tensor group_flags,
    at::Tensor group_map) {
  group_flags.zero_();
  gemm_count_t.zero_();

  compute_gemm_counts(unique_values, tensor_p_shapes, group_flags, gemm_count_t);

  int32_t part1_gemm_count = gemm_count_t.item<int32_t>(); // sync
  return part1_gemm_count;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> TT_3Order_forwrard_bag_cuda_no_reduce(
    at::Tensor unique_values,
    const std::vector<int>& tt_p_shapes, //[i1,i2,i3]
    const std::vector<int>& tt_q_shapes, //[j1,j2,j3]
    const std::vector<int>& tt_ranks, //[1,r1,r2,1]
    Tensor tensor_p_shapes, //[i1,i2,i3]
    Tensor tensor_q_shapes, //[j1,j2,j3]
    Tensor tensor_ranks, //[1,r1,r2,1]
    const std::vector<Tensor>& tt_cores,

    int32_t part1_gemm_count,
    at::Tensor gemm_count,
    at::Tensor group_flags,
    at::Tensor group_map,
    Tensor trs,
    std::optional<Tensor> batch_output) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(tt_cores[0].get_device());

  int32_t num_uni = unique_values.numel(); // unique value size

  bool check_inputs = true;
  bool need_part2 = batch_output.has_value();

  if (check_inputs) {
    CHECK_INPUT(unique_values);
    for (auto tt_core : tt_cores) {
      CHECK_INPUT(tt_core);
    }

    TORCH_CHECK(tensor_p_shapes.dtype() == at::kLong);
    TORCH_CHECK(tensor_q_shapes.dtype() == at::kLong);
    TORCH_CHECK(unique_values.dtype() == at::kLong);
  }
  TORCH_CHECK(tt_p_shapes.size() == 3ull, "Only support 3-order Tensor-train.")
  if (need_part2) {
    TORCH_CHECK(batch_output->size(0) >= num_uni)
  }

  auto num_ptrs = part1_gemm_count + (need_part2 ? num_uni : 0);

  auto a_ptr_tensor = at::empty({num_ptrs}, tt_cores[0].options().dtype(at::kLong));
  auto b_ptr_tensor = at::empty({num_ptrs}, tt_cores[0].options().dtype(at::kLong));
  auto c_ptr_tensor = at::empty({num_ptrs}, tt_cores[0].options().dtype(at::kLong));

  float** a_ptr = (float**)a_ptr_tensor.data_ptr<int64_t>();
  float** b_ptr = (float**)b_ptr_tensor.data_ptr<int64_t>();
  float** c_ptr = (float**)c_ptr_tensor.data_ptr<int64_t>();

  int32_t cache_dim = tt_q_shapes[0] * tt_q_shapes[1] * tt_ranks[2];

  {
    gemm_count.zero_();
    group_flags.zero_();

    int32_t threads = (num_uni > 256) ? 256 : 32;
    int32_t blocks = (num_uni + threads - 1) / threads;
    prepare_3cores_uidx_batched_batch_part1_gemm_pointers<<<
        blocks,
        threads,
        0,
        c10::cuda::getCurrentCUDAStream()>>>(
        num_uni,
        cache_dim,
        unique_values.data_ptr<int64_t>(),
        (const int64_t*)tensor_p_shapes.data_ptr(),
        (const int64_t*)tensor_q_shapes.data_ptr(),
        (const int64_t*)tensor_ranks.data_ptr(),
        tt_cores[0].data_ptr<float>(),
        tt_cores[1].data_ptr<float>(),
        trs.data_ptr<float>(),
        a_ptr,
        b_ptr,
        c_ptr,
        group_flags.data_ptr<int32_t>(),
        gemm_count.data_ptr<int32_t>(),
        (float**)group_map.data_ptr());

    if (need_part2) {
      int32_t tr_output_dim = tt_q_shapes[0] * tt_q_shapes[1] * tt_q_shapes[2];
      prepare_3cores_uidx_batched_batch_part2_gemm_pointers<<<
          blocks,
          threads,
          0,
          c10::cuda::getCurrentCUDAStream()>>>(
          num_uni,
          part1_gemm_count,
          tr_output_dim,
          unique_values.data_ptr<int64_t>(),
          (*batch_output).data_ptr<float>(),
          (const int64_t*)tensor_p_shapes.data_ptr(),
          (const int64_t*)tensor_q_shapes.data_ptr(),
          (const int64_t*)tensor_ranks.data_ptr(),
          tt_cores[2].data_ptr<float>(),
          a_ptr,
          b_ptr,
          c_ptr,
          (float**)group_map.data_ptr());
    }
  }

  // ======================== Batched GeMM ======================
  static float alpha = 1.0f;
  static float beta = 0.0f;

  cuda_gemm_batched_fp32_fp32(
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      tt_q_shapes[1] * tt_ranks[2], // n
      tt_q_shapes[0], // m
      tt_ranks[1], // k
      &alpha,
      (void**)a_ptr,
      tt_q_shapes[1] * tt_ranks[2], // n
      (void**)b_ptr,
      tt_ranks[1], // k
      &beta,
      (void**)c_ptr,
      tt_q_shapes[1] * tt_ranks[2], // n
      part1_gemm_count);

  // bce front contraction part2's gemm
  if (need_part2) {
    cuda_gemm_batched_fp32_fp32(
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        tt_q_shapes[2], // n
        tt_q_shapes[0] * tt_q_shapes[1], // m
        tt_ranks[2], // k
        &alpha,
        (void**)(a_ptr + part1_gemm_count),
        tt_q_shapes[2], // n
        (void**)(b_ptr + part1_gemm_count),
        tt_ranks[2], // k
        &beta,
        (void**)(c_ptr + part1_gemm_count),
        tt_q_shapes[2], // n
        num_uni);
  }

  return std::make_tuple(a_ptr_tensor, b_ptr_tensor, c_ptr_tensor);
}

std::tuple<Tensor, std::tuple<int64_t, int64_t, std::vector<int64_t>>>
TT_3Order_forward_bag_cuda_with_Uidx_batched(
    int32_t num_embeddings,
    int32_t feature_dim,
    int32_t batch_size,
    at::Tensor rowidx, // Just for checking
    at::Tensor unique_values, // Just for checking
    const std::vector<int>& tt_p_shapes, //[i1,i2,i3]
    const std::vector<int>& tt_q_shapes, //[j1,j2,j3]
    const std::vector<int>& tt_ranks, //[1,r1,r2,1]
    Tensor tensor_p_shapes, //[i1,i2,i3]
    Tensor tensor_q_shapes, //[j1,j2,j3]
    Tensor tensor_ranks, //[1,r1,r2,1]
    const std::vector<std::tuple<at::Tensor, at::Tensor, at::Tensor>>& micro_batched_inputs,
    const std::vector<Tensor>& tt_cores) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(tt_cores[0].get_device());

  bool check_inputs = true;

  if (check_inputs) {
    CHECK_INPUT(rowidx);
    CHECK_INPUT(unique_values);
    for (auto tt_core : tt_cores) {
      CHECK_INPUT(tt_core);
    }

    TORCH_CHECK(rowidx.dtype() == at::kLong)
    TORCH_CHECK(tensor_p_shapes.dtype() == at::kLong)
    TORCH_CHECK(tensor_q_shapes.dtype() == at::kLong)
  }
  TORCH_CHECK(tt_p_shapes.size() == 3ull, "Only support 3-order Tensor-train.")
  // TORCH_CHECK_EQ(micro_batched_inputs.size(), num_micros);
  TORCH_CHECK(feature_dim % 4 == 0, "feature_dim must be aligned with 4.");

  auto gemm_count_t = at::empty({1}, tt_cores[0].options().dtype(at::kInt));
  auto group_flags =
      at::empty({tt_p_shapes[0] * tt_p_shapes[1]}, tt_cores[0].options().dtype(at::kInt));
  auto group_map =
      at::empty({tt_p_shapes[0] * tt_p_shapes[1]}, tt_cores[0].options().dtype(at::kLong));

  int32_t num_micros = micro_batched_inputs.size();

  int64_t max_micro_nnz = -1;
  int64_t max_part1_gemm_cnt = -1;
  std::vector<int64_t> part1_gemm_cnts(num_micros);
  for (int32_t i = 0; i < num_micros; ++i) {
    auto micros = micro_batched_inputs[i];
    auto micro_uniques = std::get<0>(micros);

    part1_gemm_cnts[i] = compute_uidx_batch_part1_gemm_cnt(
        tt_p_shapes,
        tt_q_shapes,
        tt_ranks,
        tensor_p_shapes,
        micro_uniques,
        tt_cores,
        gemm_count_t,
        group_flags,
        group_map);

    max_part1_gemm_cnt = std::max(max_part1_gemm_cnt, part1_gemm_cnts[i]);
    max_micro_nnz = std::max(max_micro_nnz, micro_uniques.numel());
  }

  auto micro_infos = std::tuple<int64_t, int64_t, std::vector<int64_t>>(
      max_part1_gemm_cnt, max_micro_nnz, part1_gemm_cnts);

  int32_t cache_dim = tt_q_shapes[0] * tt_q_shapes[1] * tt_ranks[2];
  auto batch_trs = at::empty({max_part1_gemm_cnt, cache_dim}, tt_cores[0].options());

  int32_t prod_dim = tt_q_shapes[0] * tt_q_shapes[1] * tt_q_shapes[2];
  auto batch_output = at::empty({max_micro_nnz, prod_dim}, tt_cores[0].options());
  auto output = at::zeros({batch_size, feature_dim}, tt_cores[0].options());

  for (int32_t i = 0; i < num_micros; ++i) {
    auto& [micro_uniques, micro_back_rowidx, micro_rowidx] = micro_batched_inputs[i];
    CHECK_INPUT(micro_uniques);
    CHECK_INPUT(micro_back_rowidx);
    CHECK_INPUT(micro_rowidx);

    if (check_inputs) { // Check no copy
      TORCH_CHECK_EQ(micro_uniques.storage().data_ptr(), unique_values.storage().data_ptr());
      TORCH_CHECK_EQ(micro_rowidx.storage().data_ptr(), rowidx.storage().data_ptr());
    }

    TT_3Order_forwrard_bag_cuda_no_reduce(
        micro_uniques,
        tt_p_shapes,
        tt_q_shapes,
        tt_ranks,
        tensor_p_shapes,
        tensor_q_shapes,
        tensor_ranks,
        tt_cores,
        part1_gemm_cnts[i],
        gemm_count_t,
        group_flags,
        group_map,
        batch_trs,
        batch_output);

    int32_t N = micro_rowidx.numel();
    int32_t tx = kWarpSize;
    int32_t ty = 1024 / tx;
    dim3 threads(tx, ty);
    int32_t blocks = (N + ty - 1) / ty;
    batched_reduce_output_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        N,
        feature_dim,
        prod_dim,
        micro_rowidx.data_ptr<int64_t>(),
        micro_back_rowidx.data_ptr<int64_t>(),
        batch_output.data_ptr<float>(),
        output.data_ptr<float>());
  }

  // return output;
  return std::make_tuple(output, micro_infos);
}

at::Tensor TT_3Order_forward_bag_cuda(
    int32_t num_embeddings,
    int32_t feature_dim,
    int32_t batch_size,
    at::Tensor rowidx,
    at::Tensor unique_values,
    at::Tensor unique_keys, // unique_values -> indices
    at::Tensor unique_counts,
    at::Tensor back_map,
    at::Tensor bce_idx, // [bce_fron_idx + bce_back_idx] # unique_values[bce_map] = bce_idx
    at::Tensor bce_map, // bce_idx -> unique_values
    int32_t bce_front_num,
    const std::vector<int>& tt_p_shapes, //[i1,i2,i3]
    const std::vector<int>& tt_q_shapes, //[j1,j2,j3]
    const std::vector<int>& tt_ranks, //[1,r1,r2,1]
    Tensor tensor_p_shapes, //[i1,i2,i3]
    Tensor tensor_q_shapes, //[j1,j2,j3]
    Tensor tensor_ranks, //[1,r1,r2,1]
    const std::vector<Tensor>& tt_cores) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(tt_cores[0].get_device());

  int32_t nnz = rowidx.numel(); // batch count
  // int32_t batch_size = offsets.numel() - 1;  // batch size
  int32_t num_uni = unique_values.numel(); // unique value size

  bool check_inputs = true;

  if (check_inputs) {
    CHECK_INPUT(rowidx);
    CHECK_INPUT(unique_values);
    CHECK_INPUT(unique_keys);
    CHECK_INPUT(bce_idx);
    CHECK_INPUT(bce_map);
    for (auto tt_core : tt_cores) {
      CHECK_INPUT(tt_core);
    }

    TORCH_CHECK(rowidx.dtype() == at::kLong);
    TORCH_CHECK(tensor_p_shapes.dtype() == at::kLong);
    TORCH_CHECK(tensor_q_shapes.dtype() == at::kLong);
    TORCH_CHECK(unique_values.dtype() == at::kLong && unique_keys.dtype() == at::kLong);
    TORCH_CHECK(bce_idx.dtype() == at::kLong);
    TORCH_CHECK(bce_map.dtype() == at::kLong);
    TORCH_CHECK(
        bce_idx.numel() == unique_values.numel(),
        "bce_idx.numel() = ",
        bce_idx.numel(),
        ", unique_values.numel() = ",
        unique_values.numel());
  }
  TORCH_CHECK(tt_p_shapes.size() == 3ull, "Only support 3-order Tensor-train.")
  TORCH_CHECK(num_uni == bce_idx.numel() && num_uni == bce_map.numel());
  TORCH_CHECK(nnz == unique_keys.numel());

  auto front_gemm_count_t = at::zeros({1}, tt_cores[0].options().dtype(at::kInt));
  auto back_gemm_count_t = at::zeros({1}, tt_cores[0].options().dtype(at::kInt));
  auto front_group_flags =
      at::zeros({tt_p_shapes[0] * tt_p_shapes[1]}, tt_cores[0].options().dtype(at::kInt));
  auto front_group_map =
      at::zeros({tt_p_shapes[0] * tt_p_shapes[1]}, tt_cores[0].options().dtype(at::kLong));

  auto back_group_flags =
      at::zeros({tt_p_shapes[1] * tt_p_shapes[2]}, tt_cores[0].options().dtype(at::kInt));
  auto back_group_map =
      at::zeros({tt_p_shapes[1] * tt_p_shapes[2]}, tt_cores[0].options().dtype(at::kLong));

  compute_BCE_batch_gemm_counts(
      bce_front_num,
      num_uni,
      bce_idx,
      tensor_p_shapes,
      front_group_flags,
      back_group_flags,
      front_gemm_count_t,
      back_gemm_count_t);

  int32_t front_gemm_count = front_gemm_count_t.item<int32_t>();
  int32_t back_gemm_count = back_gemm_count_t.item<int32_t>();
  int32_t bce_part1_gemm_count = front_gemm_count + back_gemm_count;

  front_group_flags.zero_();
  back_group_flags.zero_();
  front_gemm_count_t.zero_();
  back_gemm_count_t.zero_();

  auto a_ptr_tensor = at::empty(
      {front_gemm_count + back_gemm_count + num_uni}, tt_cores[0].options().dtype(at::kLong));
  auto b_ptr_tensor = at::empty(
      {front_gemm_count + back_gemm_count + num_uni}, tt_cores[0].options().dtype(at::kLong));
  auto c_ptr_tensor = at::empty(
      {front_gemm_count + back_gemm_count + num_uni}, tt_cores[0].options().dtype(at::kLong));

  float** a_ptr = (float**)a_ptr_tensor.data_ptr<int64_t>();
  float** b_ptr = (float**)b_ptr_tensor.data_ptr<int64_t>();
  float** c_ptr = (float**)c_ptr_tensor.data_ptr<int64_t>();

  auto output = at::zeros({batch_size, feature_dim}, tt_cores[0].options());

  int32_t front_cache_dim = tt_q_shapes[0] * tt_q_shapes[1] * tt_ranks[2];
  auto front_trs = at::empty({front_gemm_count, front_cache_dim}, tt_cores[0].options());

  int32_t back_cache_dim = tt_q_shapes[1] * tt_q_shapes[2] * tt_ranks[1];
  auto back_trs = at::empty({back_gemm_count, back_cache_dim}, tt_cores[0].options());

  int32_t tr_output_dim = tt_q_shapes[0] * tt_q_shapes[1] * tt_q_shapes[2];
  auto uni_output = at::empty({num_uni, tr_output_dim}, tt_cores[0].options());

  {
    int32_t threads = (num_uni > 256) ? 256 : 32;
    int32_t blocks = (num_uni + threads - 1) / threads;
    bce_prepare_part1_batch_gemm_pointers<<<
        blocks,
        threads,
        0,
        c10::cuda::getCurrentCUDAStream()>>>(
        num_uni,
        bce_front_num,
        front_gemm_count,
        front_cache_dim,
        back_cache_dim,
        bce_idx.data_ptr<int64_t>(),
        bce_map.data_ptr<int64_t>(),
        (const int64_t*)tensor_p_shapes.data_ptr(),
        (const int64_t*)tensor_q_shapes.data_ptr(),
        (const int64_t*)tensor_ranks.data_ptr(),
        tt_cores[0].data_ptr<float>(),
        tt_cores[1].data_ptr<float>(),
        tt_cores[2].data_ptr<float>(),
        front_trs.data_ptr<float>(),
        back_trs.data_ptr<float>(),
        a_ptr,
        b_ptr,
        c_ptr,
        front_group_flags.data_ptr<int32_t>(),
        back_group_flags.data_ptr<int32_t>(),
        front_gemm_count_t.data_ptr<int32_t>(),
        back_gemm_count_t.data_ptr<int32_t>(),
        (float**)front_group_map.data_ptr(),
        (float**)back_group_map.data_ptr());

    bce_prepare_part2_batch_gemm_pointers<<<
        blocks,
        threads,
        0,
        c10::cuda::getCurrentCUDAStream()>>>(
        num_uni,
        bce_front_num,
        tr_output_dim,
        bce_part1_gemm_count,
        bce_idx.data_ptr<int64_t>(),
        bce_map.data_ptr<int64_t>(),
        (const int64_t*)tensor_p_shapes.data_ptr(),
        (const int64_t*)tensor_q_shapes.data_ptr(),
        (const int64_t*)tensor_ranks.data_ptr(),
        tt_cores[0].data_ptr<float>(),
        tt_cores[2].data_ptr<float>(),
        uni_output.data_ptr<float>(),
        a_ptr,
        b_ptr,
        c_ptr,
        (float**)front_group_map.data_ptr(),
        (float**)back_group_map.data_ptr());
  }
  // std::cout << (a_ptr_tensor) << std::endl;
  // std::cout << (b_ptr_tensor) << std::endl;
  // std::cout << (c_ptr_tensor) << std::endl;

  // print_ptr<<<1, 1>>>(a_ptr, b_ptr, c_ptr);

  // ======================== Batched GeMM ======================
  static float alpha = 1.0f;
  static float beta = 0.0f;

  // bce front contraction part1's
  cuda_gemm_batched_fp32_fp32(
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      tt_q_shapes[1] * tt_ranks[2], // n
      tt_q_shapes[0], // m
      tt_ranks[1], // k
      &alpha,
      (void**)a_ptr,
      tt_q_shapes[1] * tt_ranks[2], // n
      (void**)b_ptr,
      tt_ranks[1], // k
      &beta,
      (void**)c_ptr,
      tt_q_shapes[1] * tt_ranks[2], // n
      front_gemm_count);

  // bce back contraction part1's
  cuda_gemm_batched_fp32_fp32(
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      tt_q_shapes[2], // n
      tt_q_shapes[1] * tt_ranks[1], // m
      tt_ranks[2], // k
      &alpha,
      (void**)(a_ptr + front_gemm_count),
      tt_q_shapes[2], // n
      (void**)(b_ptr + front_gemm_count),
      tt_ranks[2], // k
      &beta,
      (void**)(c_ptr + front_gemm_count),
      tt_q_shapes[2], // n
      back_gemm_count);

  // bce front contraction part2's gemm
  cuda_gemm_batched_fp32_fp32(
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      tt_q_shapes[2], // n
      tt_q_shapes[0] * tt_q_shapes[1], // m
      tt_ranks[2], // k
      &alpha,
      (void**)(a_ptr + bce_part1_gemm_count),
      tt_q_shapes[2], // n
      (void**)(b_ptr + bce_part1_gemm_count),
      tt_ranks[2], // k
      &beta,
      (void**)(c_ptr + bce_part1_gemm_count),
      tt_q_shapes[2], // n
      bce_front_num);

  // bce back contraction part2's gemm
  cuda_gemm_batched_fp32_fp32(
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      tt_q_shapes[1] * tt_q_shapes[2], // n
      tt_q_shapes[0], // m
      tt_ranks[1], // k
      &alpha,
      (void**)(a_ptr + bce_part1_gemm_count + bce_front_num),
      tt_q_shapes[1] * tt_q_shapes[2], // n
      (void**)(b_ptr + bce_part1_gemm_count + bce_front_num),
      tt_ranks[1], // k
      &beta,
      (void**)(c_ptr + bce_part1_gemm_count + bce_front_num),
      tt_q_shapes[1] * tt_q_shapes[2], // n
      num_uni - bce_front_num);

  // Pre-reduce, avoid to allocate intermidate for all indices embedding.
  {
    int32_t tx = kWarpSize;
    int32_t ty = 1024 / tx;
    dim3 threads(tx, ty);
    int32_t blocks = (nnz + ty - 1) / ty;
    reduce_output_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        nnz,
        feature_dim,
        tr_output_dim,
        rowidx.data_ptr<int64_t>(),
        unique_keys.data_ptr<int64_t>(),
        uni_output.data_ptr<float>(),
        output.data_ptr<float>());
  }
  return output;
}

__global__ void compute_rowidx_kernel(
    int32_t B,
    const int64_t* __restrict__ offsets,
    int64_t* __restrict__ rowidx) {
  int32_t b = blockIdx.x * blockDim.y + threadIdx.y;
  // printf("b < B:%d\n", b < B);
  if (b < B) {
    int64_t colidx_start = offsets[b];
    int64_t colidx_end = offsets[b + 1];
    int32_t L = colidx_end - colidx_start;
    for (int32_t l = threadIdx.x; l < L; l += blockDim.x) {
      rowidx[l + colidx_start] = b;
    }
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> enhance_unique(at::Tensor x) {
  CHECK_INPUT(x);
  TORCH_CHECK(x.dtype() == at::kLong);
  auto [sorted, back_map] = at::sort(x);
  auto [uniques, inverse, counts] = at::unique_consecutive(sorted, true, true);

  return std::make_tuple(uniques, inverse, counts, back_map);
}

at::Tensor compute_rowidx_cuda(Tensor indices, Tensor offsets) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(indices.get_device());

  CHECK_INPUT(indices);
  CHECK_INPUT(offsets);

  TORCH_CHECK(indices.dtype() == at::kLong)
  auto rowidx = at::empty_like(indices);

  int32_t B = offsets.numel() - 1;

  int32_t tx = 8;
  int32_t ty = 32;
  compute_rowidx_kernel<<<
      div_round_up(B, ty),
      dim3(tx, ty),
      0,
      c10::cuda::getCurrentCUDAStream()>>>(
      B, offsets.data_ptr<int64_t>(), rowidx.data_ptr<int64_t>());

  return rowidx;
}

//=============================================================================================================================

__global__ void Extra_Eff_prepare_batch_gemm_pointers_3_core_backward(
    int32_t unique_num,
    const int64_t* __restrict__ unique_index,
    const int64_t* tt_p_shape,

    PackedTensorAccessor64<float, 2, RestrictPtrTraits> tt_cores_0,
    PackedTensorAccessor64<float, 2, RestrictPtrTraits> tt_cores_1,
    PackedTensorAccessor64<float, 2, RestrictPtrTraits> tt_cores_2,
    PackedTensorAccessor64<float, 2, RestrictPtrTraits> tr_tt_cores_0,
    PackedTensorAccessor64<float, 2, RestrictPtrTraits> tr_tt_cores_1,
    PackedTensorAccessor64<float, 2, RestrictPtrTraits> tr_tt_cores_2,
    PackedTensorAccessor64<float, 2, RestrictPtrTraits> tr_0,
    PackedTensorAccessor64<float, 2, RestrictPtrTraits> d_output,
    int32_t* __restrict__ tt_idx,
    float** __restrict__ a_ptr,
    float** __restrict__ b_ptr,
    float** __restrict__ c_ptr,
    float** __restrict__ a0_ptr,
    float** __restrict__ b0_ptr,
    float** __restrict__ c0_ptr,
    float** __restrict__ a1_ptr,
    float** __restrict__ b1_ptr,
    float** __restrict__ c1_ptr) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  assert(0);
  if (n < unique_num) {
    int32_t idx = *(unique_index + n);

    float tmp = float(idx) / tt_p_shape[2];
    int group = floor(tmp);
    int I3 = idx % tt_p_shape[2];
    int I1 = floor(float(group) / tt_p_shape[1]);
    int I2 = group % tt_p_shape[1];

    tt_idx[0 * unique_num + n] = I1;
    tt_idx[1 * unique_num + n] = I2;
    tt_idx[2 * unique_num + n] = I3;
    float* tr_0_ptr = (float*)&(tr_0[n][0]);
    float* d_output_ptr = (float*)&(d_output[n][0]);
    float* tt_cores_0_ptr = (float*)&(tt_cores_0[I1][0]);
    float* tt_cores_1_ptr = (float*)&(tt_cores_1[I2][0]);
    a_ptr[0 * unique_num + n] = tt_cores_1_ptr; // T1*T2
    b_ptr[0 * unique_num + n] = tt_cores_0_ptr;
    c_ptr[0 * unique_num + n] = tr_0_ptr;

    a0_ptr[1 * unique_num + n] = tr_0_ptr;
    b0_ptr[1 * unique_num + n] = d_output_ptr;
    c0_ptr[1 * unique_num + n] = (float*)&(tr_tt_cores_2[n][0]);
    a1_ptr[1 * unique_num + n] = d_output_ptr;
    b1_ptr[1 * unique_num + n] = (float*)&(tt_cores_2[I3][0]);
    c1_ptr[1 * unique_num + n] = tr_0_ptr;

    a0_ptr[0 * unique_num + n] = tt_cores_0_ptr;
    b0_ptr[0 * unique_num + n] = tr_0_ptr;
    c0_ptr[0 * unique_num + n] = (float*)&(tr_tt_cores_1[n][0]);
    a1_ptr[0 * unique_num + n] = tr_0_ptr;
    b1_ptr[0 * unique_num + n] = tt_cores_1_ptr;
    c1_ptr[0 * unique_num + n] = (float*)&(tr_tt_cores_0[n][0]);
  }
}

__global__ void compute_unique_gradient(
    int batch_size,
    int feature_dim,
    const int64_t* inverse,
    PackedTensorAccessor64<float, 2, RestrictPtrTraits> d_input,
    PackedTensorAccessor64<float, 2, RestrictPtrTraits> d_output) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= batch_size)
    return;

  int idx = inverse[n];
  for (int i = 0; i < feature_dim; i++) {
    atomicAdd(&(d_output[idx][i]), d_input[n][i]);
  }
}

__global__ void extra_fused_update_tt_cores_sgd_kernel(
    int32_t B,
    int32_t D,
    float learning_rate,
    const int32_t* __restrict__ tt_idx,
    PackedTensorAccessor64<float, 2, RestrictPtrTraits> tr_tt_cores,
    PackedTensorAccessor64<float, 2, RestrictPtrTraits> tt_core) {
  int32_t n = blockIdx.x * blockDim.y + threadIdx.y;
  if (n >= B) {
    return;
  }
  auto idx = __ldg(&tt_idx[n]);
  for (int32_t d = threadIdx.x; d < D; d += blockDim.x) {
    auto delta = -learning_rate * tr_tt_cores[n][d];
    atomicAdd(&(tt_core[idx][d]), delta);
    // tt_core[idx][d] -= learning_rate * tr_tt_cores[n][d];
  }
}

void Fused_Extra_Efficient_TT_backward_sgd_cuda(
    int32_t batch_size,
    int32_t table_length,
    int32_t feature_dim,
    float learning_rate,

    const Tensor index,
    const std::vector<int>& tt_p_shapes, //[i1,i2,i3]
    const std::vector<int>& tt_q_shapes, //[j1,j2,j3]
    const std::vector<int>& tt_ranks, //[1,r1,r2,1]
    const Tensor tensor_p_shapes, //[i1,i2,i3]
    const Tensor tensor_q_shapes, //[j1,j2,j3]
    const Tensor tensor_ranks, //[1,r1,r2,1]
    Tensor d_output,
    std::vector<Tensor>& tt_cores,
    Tensor sorted_idx,
    Tensor sorted_key) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(d_output.get_device());
  int32_t T = 3; // 3
  int32_t batch_count = batch_size;
  int32_t N = batch_size;
  int32_t unique_num = sorted_idx.size(0);

  auto unique_d = at::zeros({sorted_idx.size(0), d_output.size(1)}, tt_cores[0].options());

  int32_t threads = (N > 256 ? 256 : 32);
  int32_t num_blocks = (N + threads - 1) / threads;

  compute_unique_gradient<<<num_blocks, threads>>>(
      batch_size,
      feature_dim,
      (const int64_t*)sorted_key.data_ptr(),
      d_output.packed_accessor64<float, 2, RestrictPtrTraits>(),
      unique_d.packed_accessor64<float, 2, RestrictPtrTraits>());

  //===================================================================================================================
  std::vector<Tensor> tr_tt_cores;
  tr_tt_cores.push_back(at::empty({unique_num, tt_cores[0].size(1)}, tt_cores[0].options()));
  tr_tt_cores.push_back(at::empty({unique_num, tt_cores[1].size(1)}, tt_cores[1].options()));
  tr_tt_cores.push_back(at::empty({unique_num, tt_cores[2].size(1)}, tt_cores[2].options()));

  std::vector<int32_t> m(T - 1);
  std::vector<int32_t> n(T - 1);
  std::vector<int32_t> k(T - 1);
  float alpha = 1.0;
  float beta = 0.0;
  int32_t m_ = tt_q_shapes[0];
  for (int32_t t = 0; t < T - 1; ++t) {
    m[t] = m_; // m[0]=j1 m[1]=j1*j2
    k[t] = tt_ranks[t + 1]; // k[0]=r1 k[1]=r2
    n[t] = tt_q_shapes[t + 1] * tt_ranks[t + 2]; // n[0]=j2*r2 n[1]=j3
    m_ = m_ * tt_q_shapes[t + 1];
  }

  std::vector<Tensor> tr;

  int64_t tr_size = tt_q_shapes[0] * tt_ranks[1];
  for (int32_t t = 0; t < T - 2; ++t) {
    tr_size = tr_size * tt_q_shapes[t + 1] * tt_ranks[t + 2] / tt_ranks[t + 1];
    tr.push_back(at::empty({unique_num, tr_size}, tt_cores[0].options()));
  }

  auto tt_idx = at::empty({T * unique_num}, tt_cores[0].options().dtype(at::kInt));
  auto a_ptr_tensor = at::empty({(T - 2) * unique_num}, tt_cores[0].options().dtype(at::kLong));
  auto b_ptr_tensor = at::empty({(T - 2) * unique_num}, tt_cores[0].options().dtype(at::kLong));
  auto c_ptr_tensor = at::empty({(T - 2) * unique_num}, tt_cores[0].options().dtype(at::kLong));
  float** a_ptr = (float**)a_ptr_tensor.data_ptr<int64_t>();
  float** b_ptr = (float**)b_ptr_tensor.data_ptr<int64_t>();
  float** c_ptr = (float**)c_ptr_tensor.data_ptr<int64_t>();
  auto a0_ptr_tensor = at::empty({(T - 1) * unique_num}, tt_cores[0].options().dtype(at::kLong));
  auto b0_ptr_tensor = at::empty({(T - 1) * unique_num}, tt_cores[0].options().dtype(at::kLong));
  auto c0_ptr_tensor = at::empty({(T - 1) * unique_num}, tt_cores[0].options().dtype(at::kLong));
  float** a0_ptr = (float**)a0_ptr_tensor.data_ptr<int64_t>();
  float** b0_ptr = (float**)b0_ptr_tensor.data_ptr<int64_t>();
  float** c0_ptr = (float**)c0_ptr_tensor.data_ptr<int64_t>();
  auto a1_ptr_tensor = at::empty({(T - 1) * unique_num}, tt_cores[0].options().dtype(at::kLong));
  auto b1_ptr_tensor = at::empty({(T - 1) * unique_num}, tt_cores[0].options().dtype(at::kLong));
  auto c1_ptr_tensor = at::empty({(T - 1) * unique_num}, tt_cores[0].options().dtype(at::kLong));
  float** a1_ptr = (float**)a1_ptr_tensor.data_ptr<int64_t>();
  float** b1_ptr = (float**)b1_ptr_tensor.data_ptr<int64_t>();
  float** c1_ptr = (float**)c1_ptr_tensor.data_ptr<int64_t>();

  threads = (unique_num > 256 ? 256 : 32);
  num_blocks = (unique_num + threads - 1) / threads;

  Extra_Eff_prepare_batch_gemm_pointers_3_core_backward<<<
      num_blocks,
      threads,
      0,
      c10::cuda::getCurrentCUDAStream()>>>(
      unique_num,
      (const int64_t*)sorted_idx.data_ptr(),
      (const int64_t*)tensor_p_shapes.data_ptr(),

      tt_cores[0].packed_accessor64<float, 2, RestrictPtrTraits>(),
      tt_cores[1].packed_accessor64<float, 2, RestrictPtrTraits>(),
      tt_cores[2].packed_accessor64<float, 2, RestrictPtrTraits>(),

      tr_tt_cores[0].packed_accessor64<float, 2, RestrictPtrTraits>(),
      tr_tt_cores[1].packed_accessor64<float, 2, RestrictPtrTraits>(),
      tr_tt_cores[2].packed_accessor64<float, 2, RestrictPtrTraits>(),
      tr[0].packed_accessor64<float, 2, RestrictPtrTraits>(),
      unique_d.packed_accessor64<float, 2, RestrictPtrTraits>(),
      tt_idx.data_ptr<int32_t>(),
      a_ptr,
      b_ptr,
      c_ptr,
      a0_ptr,
      b0_ptr,
      c0_ptr,
      a1_ptr,
      b1_ptr,
      c1_ptr);

  cuda_gemm_batched_fp32_fp32(
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      n[0],
      m[0],
      k[0],
      &alpha,
      (void**)&(a_ptr[0]),
      n[0],
      (void**)&(b_ptr[0]),
      k[0],
      &beta,
      (void**)&(c_ptr[0]),
      n[0],
      unique_num);

  // //======================================================
  cuda_gemm_batched_fp32_fp32(
      CUBLAS_OP_N,
      CUBLAS_OP_T,
      n[1],
      k[1],
      m[1],
      &alpha,
      (void**)&(b0_ptr[unique_num]),
      n[1],
      (void**)&(a0_ptr[unique_num]),
      k[1],
      &beta,
      (void**)&(c0_ptr[unique_num]),
      n[1],
      unique_num);

  cuda_gemm_batched_fp32_fp32(
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      k[1],
      m[1],
      n[1],
      &alpha,
      (void**)&(b1_ptr[unique_num]),
      n[1],
      (void**)&(a1_ptr[unique_num]),
      n[1],
      &beta,
      (void**)&(c1_ptr[unique_num]),
      k[1],
      unique_num);
  //=========================================================

  cuda_gemm_batched_fp32_fp32(
      CUBLAS_OP_N,
      CUBLAS_OP_T,
      n[0],
      k[0],
      m[0],
      &alpha,
      (void**)&(b0_ptr[0]),
      n[0],
      (void**)&(a0_ptr[0]),
      k[0],
      &beta,
      (void**)&(c0_ptr[0]),
      n[0],
      unique_num);

  cuda_gemm_batched_fp32_fp32(
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      k[0],
      m[0],
      n[0],
      &alpha,
      (void**)&(b1_ptr[0]),
      n[0],
      (void**)&(a1_ptr[0]),
      n[0],
      &beta,
      (void**)&(c1_ptr[0]),
      k[0],
      unique_num);

  //=========================================================
  // return (a + b - 1) / b;
  for (int32_t t = 0; t < T; ++t) {
    int32_t D_0 = tt_cores[t].size(1);
    int32_t tx_0 = std::min(1024, D_0);
    int32_t ty_0 = 1024 / tx_0;
    extra_fused_update_tt_cores_sgd_kernel<<<
        div_round_up(unique_num, ty_0),
        dim3(tx_0, ty_0),
        0,
        c10::cuda::getCurrentCUDAStream()>>>(
        unique_num,
        D_0,
        learning_rate,
        &(tt_idx.data_ptr<int32_t>()[t * unique_num]),
        tr_tt_cores[t].packed_accessor64<float, 2, RestrictPtrTraits>(),
        tt_cores[t].packed_accessor64<float, 2, RestrictPtrTraits>());
  }
  return;
}

__global__ void generate_d_output_kernel(
    int32_t N, // batch cnt
    int32_t B, // batch_size
    int32_t D, // feature dim
    const int64_t* __restrict__ rowidx,
    const float* __restrict__ d_reduced_output,
    float* __restrict__ d_output) {
  int32_t n = blockIdx.x * blockDim.x + threadIdx.x;

  if (n >= N) {
    return;
  }

  int32_t idx = rowidx[n];
  assert(idx < B);

  for (int32_t d = 0; d < D; ++d) {
    d_output[n * D + d] = d_reduced_output[idx * D + d];
    // atomicAdd(&(d_output[n * D + d]), d_reduced_output[idx * D + d]);
  }
}

__global__ void aggregate_gradient_kernel(
    int32_t N, // batch cnt
    int32_t D, // feature dim
    int32_t tr_output_dim, // tr_output_dim
    const int64_t* __restrict__ rowidx,
    const int64_t* __restrict__ unique_keys,
    const float* __restrict__ reduced_d,
    float* __restrict__ uni_d) {
  int32_t indice_id = blockIdx.x * blockDim.y + threadIdx.y;
  if (indice_id >= N) {
    // don't have *warp* divergence since we launch full warps in blockDim.x,
    // so we can just exit this warp entirely.
    return;
  }
  // check if this warp is responsible for this whole segment.
  bool segment_start = (indice_id == 0 || rowidx[indice_id - 1] != rowidx[indice_id]);
  if (!segment_start) {
    // don't have *warp* divergence since we launch full warps in blockDim.x,
    // so we can just exit this warp entirely.
    return;
  }
  int64_t reduced_idx = rowidx[indice_id];
  // now, find the end of the segment (and thus the segment length `SL`).
  int32_t SL = 1;
  while (indice_id + SL < N && rowidx[indice_id + SL] == reduced_idx) {
    SL += 1;
  }
  for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
    Vec4T<float> grads(&reduced_d[reduced_idx * D + d * 4]);
    for (int32_t sl = 0; sl < SL; ++sl) {
      int64_t uni_idx = __ldg(&unique_keys[indice_id + sl]);
      atomicAdd(&(uni_d[uni_idx * tr_output_dim + d * 4]), grads.acc.x);
      atomicAdd(&(uni_d[uni_idx * tr_output_dim + d * 4 + 1]), grads.acc.y);
      atomicAdd(&(uni_d[uni_idx * tr_output_dim + d * 4 + 2]), grads.acc.z);
      atomicAdd(&(uni_d[uni_idx * tr_output_dim + d * 4 + 3]), grads.acc.w);
    }
  }
}

__global__ void aggregate_gradient_kernel_v2(
    int32_t N, // batch cnt
    int32_t D, // feature dim for reduced_d
    int32_t tr_output_dim, // for uni_d
    const int64_t* __restrict__ rowidx,
    const int64_t* __restrict__ unique_keys,
    const float* __restrict__ reduced_d,
    float* __restrict__ uni_d) {
  int32_t indice_id = blockIdx.x * blockDim.y + threadIdx.y;
  if (indice_id >= N) {
    return;
  }

  int64_t reduced_idx = __ldg(&rowidx[indice_id]);
  int64_t uni_idx = __ldg(&unique_keys[indice_id]);
  for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
    Vec4T<float> grads(&reduced_d[reduced_idx * D + d * 4]);
    atomicAdd(&(uni_d[uni_idx * tr_output_dim + d * 4]), grads.acc.x);
    atomicAdd(&(uni_d[uni_idx * tr_output_dim + d * 4 + 1]), grads.acc.y);
    atomicAdd(&(uni_d[uni_idx * tr_output_dim + d * 4 + 2]), grads.acc.z);
    atomicAdd(&(uni_d[uni_idx * tr_output_dim + d * 4 + 3]), grads.acc.w);
  }
}

at::Tensor aggregate_gradients(
    int32_t nnz, // batch count
    int32_t num_uni,
    int32_t tr_output_dim,
    int32_t feature_dim,
    at::Tensor rowidx,
    at::Tensor unique_keys,
    at::Tensor reduced_d) {
  auto unique_d = at::zeros({num_uni, tr_output_dim}, reduced_d.options());

  int32_t tx = kWarpSize;
  int32_t ty = 1024 / tx;
  dim3 threads(tx, ty);
  int32_t blocks = (nnz + ty - 1) / ty;
  aggregate_gradient_kernel_v2<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      nnz,
      feature_dim,
      tr_output_dim,
      rowidx.data_ptr<int64_t>(),
      unique_keys.data_ptr<int64_t>(),
      reduced_d.data_ptr<float>(),
      unique_d.data_ptr<float>());

  return unique_d;
}

void aggregate_gradients(
    at::Tensor inplace,
    int32_t nnz, // batch count
    int32_t num_uni,
    int32_t tr_output_dim,
    int32_t feature_dim,
    at::Tensor rowidx,
    at::Tensor unique_keys,
    at::Tensor reduced_d) {
  inplace.zero_();

  int32_t tx = kWarpSize;
  int32_t ty = 1024 / tx;
  dim3 threads(tx, ty);
  int32_t blocks = (nnz + ty - 1) / ty;
  aggregate_gradient_kernel_v2<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      nnz,
      feature_dim,
      tr_output_dim,
      rowidx.data_ptr<int64_t>(),
      unique_keys.data_ptr<int64_t>(),
      reduced_d.data_ptr<float>(),
      inplace.data_ptr<float>());
}

at::Tensor aggregate_gradients_v1(
    int32_t N, // batch count
    int32_t B, // batch size
    int32_t feature_dim,
    at::Tensor d_reduced_output,
    at::Tensor rowidx,
    at::Tensor sorted_idx,
    at::Tensor sorted_key) {
  auto options = d_reduced_output.options();

  auto d_output = at::empty({N, feature_dim}, options);
  auto unique_d = at::zeros({sorted_idx.size(0), d_output.size(1)}, options);

  // *************** generate d_output *******************
  int32_t threads = (N > 256 ? 256 : 32);
  int32_t num_blocks = (N + threads - 1) / threads;
  generate_d_output_kernel<<<num_blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      N,
      B,
      feature_dim,
      rowidx.data_ptr<int64_t>(),
      d_reduced_output.data_ptr<float>(),
      d_output.data_ptr<float>());

  // std::cout << "d_output numel()" << d_output.numel() << std::endl;
  // std::cout << "unique_d numel()" << unique_d.numel() << std::endl;

  // *************** geneate unique_d *******************
  threads = (N > 256 ? 256 : 32);
  num_blocks = (N + threads - 1) / threads;
  compute_unique_gradient<<<num_blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      N,
      feature_dim,
      (const int64_t*)sorted_key.data_ptr(),
      d_output.packed_accessor64<float, 2, RestrictPtrTraits>(),
      unique_d.packed_accessor64<float, 2, RestrictPtrTraits>());

  return unique_d;
}

void Fused_TT_3Order_backward_bag_cuda(
    int32_t num_embeddings,
    int32_t feature_dim,
    float learning_rate,
    at::Tensor rowidx,
    at::Tensor unique_values,
    at::Tensor unique_keys,
    at::Tensor bce_idx,
    at::Tensor bce_map,
    int32_t bce_front_num,
    const std::vector<int>& tt_p_shapes, //[i1,i2,i3]
    const std::vector<int>& tt_q_shapes, //[j1,j2,j3]
    const std::vector<int>& tt_ranks, //[1,r1,r2,1]
    at::Tensor tensor_p_shapes, //[i1,i2,i3]
    at::Tensor tensor_q_shapes, //[j1,j2,j3]
    at::Tensor tensor_ranks, //[1,r1,r2,1]
    std::vector<Tensor>& tt_cores,
    at::Tensor reduced_d) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(reduced_d.get_device());

  int32_t nnz = rowidx.numel();
  int32_t T = 3;
  // int32_t batch_size = offsets.numel() - 1;  // batch size
  int32_t num_uni = unique_values.numel(); // unique value size
  auto fp_options = reduced_d.options();

  bool check_inputs = true;
  if (check_inputs) {
    CHECK_INPUT(rowidx);
    CHECK_INPUT(unique_values);
    CHECK_INPUT(unique_keys);
    CHECK_INPUT(bce_idx);
    CHECK_INPUT(bce_map);
    CHECK_INPUT(tensor_p_shapes);
    CHECK_INPUT(tensor_q_shapes);
    CHECK_INPUT(tensor_ranks);
    CHECK_INPUT(reduced_d);
    for (auto t : tt_cores) {
      CHECK_INPUT(t);
      TORCH_CHECK(t.device() == fp_options.device());
      TORCH_CHECK(t.dtype() == fp_options.dtype());
    }
    TORCH_CHECK(rowidx.dtype() == at::kLong)
    TORCH_CHECK(tensor_p_shapes.dtype() == at::kLong)
    TORCH_CHECK(tensor_q_shapes.dtype() == at::kLong)
    TORCH_CHECK(unique_values.dtype() == at::kLong && unique_keys.dtype() == at::kLong);
    TORCH_CHECK(bce_idx.dtype() == at::kLong)
    TORCH_CHECK(bce_map.dtype() == at::kLong)
  }
  TORCH_CHECK(tt_p_shapes.size() == 3ull, "Only support 3-order Tensor-train.")
  TORCH_CHECK(nnz == unique_keys.numel());

  int32_t tr_output_dim = tt_q_shapes[0] * tt_q_shapes[1] * tt_q_shapes[2];

  auto unique_d =
      aggregate_gradients(nnz, num_uni, tr_output_dim, feature_dim, rowidx, unique_keys, reduced_d);

  //===================================================================================================================
  std::vector<Tensor> tr_tt_cores;
  tr_tt_cores.push_back(at::empty({num_uni, tt_cores[0].size(1)}, fp_options));
  tr_tt_cores.push_back(at::empty({num_uni, tt_cores[1].size(1)}, fp_options)); // Huge!!!
  tr_tt_cores.push_back(at::empty({num_uni, tt_cores[2].size(1)}, fp_options));

  std::vector<int32_t> m(T - 1);
  std::vector<int32_t> n(T - 1);
  std::vector<int32_t> k(T - 1);
  float alpha = 1.0;
  float beta = 0.0;
  int32_t m_ = tt_q_shapes[0];
  for (int32_t t = 0; t < T - 1; ++t) {
    m[t] = m_; // m[0]=j1 m[1]=j1*j2
    k[t] = tt_ranks[t + 1]; // k[0]=r1 k[1]=r2
    n[t] = tt_q_shapes[t + 1] * tt_ranks[t + 2]; // n[0]=j2*r2 n[1]=j3
    m_ = m_ * tt_q_shapes[t + 1];
  }

  std::vector<Tensor> tr;

  int64_t tr_size = tt_q_shapes[0] * tt_ranks[1];
  for (int32_t t = 0; t < T - 2; ++t) {
    tr_size = tr_size * tt_q_shapes[t + 1] * tt_ranks[t + 2] / tt_ranks[t + 1];
    tr.push_back(at::empty({num_uni, tr_size}, tt_cores[0].options()));
  }

  auto tt_idx = at::empty({T * num_uni}, fp_options.dtype(at::kInt));
  auto a_ptr_tensor = at::empty({(T - 2) * num_uni}, tt_cores[0].options().dtype(at::kLong));
  auto b_ptr_tensor = at::empty({(T - 2) * num_uni}, tt_cores[0].options().dtype(at::kLong));
  auto c_ptr_tensor = at::empty({(T - 2) * num_uni}, tt_cores[0].options().dtype(at::kLong));
  float** a_ptr = (float**)a_ptr_tensor.data_ptr<int64_t>();
  float** b_ptr = (float**)b_ptr_tensor.data_ptr<int64_t>();
  float** c_ptr = (float**)c_ptr_tensor.data_ptr<int64_t>();
  auto a0_ptr_tensor = at::empty({(T - 1) * num_uni}, tt_cores[0].options().dtype(at::kLong));
  auto b0_ptr_tensor = at::empty({(T - 1) * num_uni}, tt_cores[0].options().dtype(at::kLong));
  auto c0_ptr_tensor = at::empty({(T - 1) * num_uni}, tt_cores[0].options().dtype(at::kLong));
  float** a0_ptr = (float**)a0_ptr_tensor.data_ptr<int64_t>();
  float** b0_ptr = (float**)b0_ptr_tensor.data_ptr<int64_t>();
  float** c0_ptr = (float**)c0_ptr_tensor.data_ptr<int64_t>();
  auto a1_ptr_tensor = at::empty({(T - 1) * num_uni}, tt_cores[0].options().dtype(at::kLong));
  auto b1_ptr_tensor = at::empty({(T - 1) * num_uni}, tt_cores[0].options().dtype(at::kLong));
  auto c1_ptr_tensor = at::empty({(T - 1) * num_uni}, tt_cores[0].options().dtype(at::kLong));
  float** a1_ptr = (float**)a1_ptr_tensor.data_ptr<int64_t>();
  float** b1_ptr = (float**)b1_ptr_tensor.data_ptr<int64_t>();
  float** c1_ptr = (float**)c1_ptr_tensor.data_ptr<int64_t>();

  {
    int32_t threads = (num_uni > 256 ? 256 : 32);
    int32_t num_blocks = (num_uni + threads - 1) / threads;

    Extra_Eff_prepare_batch_gemm_pointers_3_core_backward<<<
        num_blocks,
        threads,
        0,
        c10::cuda::getCurrentCUDAStream()>>>(
        num_uni,
        (const int64_t*)unique_values.data_ptr(),
        (const int64_t*)tensor_p_shapes.data_ptr(),

        tt_cores[0].packed_accessor64<float, 2, RestrictPtrTraits>(),
        tt_cores[1].packed_accessor64<float, 2, RestrictPtrTraits>(),
        tt_cores[2].packed_accessor64<float, 2, RestrictPtrTraits>(),

        tr_tt_cores[0].packed_accessor64<float, 2, RestrictPtrTraits>(),
        tr_tt_cores[1].packed_accessor64<float, 2, RestrictPtrTraits>(),
        tr_tt_cores[2].packed_accessor64<float, 2, RestrictPtrTraits>(),
        tr[0].packed_accessor64<float, 2, RestrictPtrTraits>(),
        unique_d.packed_accessor64<float, 2, RestrictPtrTraits>(),
        tt_idx.data_ptr<int32_t>(),
        a_ptr,
        b_ptr,
        c_ptr,
        a0_ptr,
        b0_ptr,
        c0_ptr,
        a1_ptr,
        b1_ptr,
        c1_ptr);
  }

  cuda_gemm_batched_fp32_fp32(
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      n[0],
      m[0],
      k[0],
      &alpha,
      (void**)&(a_ptr[0]),
      n[0],
      (void**)&(b_ptr[0]),
      k[0],
      &beta,
      (void**)&(c_ptr[0]),
      n[0],
      num_uni);

  // //======================================================
  cuda_gemm_batched_fp32_fp32(
      CUBLAS_OP_N,
      CUBLAS_OP_T,
      n[1],
      k[1],
      m[1],
      &alpha,
      (void**)&(b0_ptr[num_uni]),
      n[1],
      (void**)&(a0_ptr[num_uni]),
      k[1],
      &beta,
      (void**)&(c0_ptr[num_uni]),
      n[1],
      num_uni);

  cuda_gemm_batched_fp32_fp32(
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      k[1],
      m[1],
      n[1],
      &alpha,
      (void**)&(b1_ptr[num_uni]),
      n[1],
      (void**)&(a1_ptr[num_uni]),
      n[1],
      &beta,
      (void**)&(c1_ptr[num_uni]),
      k[1],
      num_uni);
  //=========================================================

  cuda_gemm_batched_fp32_fp32(
      CUBLAS_OP_N,
      CUBLAS_OP_T,
      n[0],
      k[0],
      m[0],
      &alpha,
      (void**)&(b0_ptr[0]),
      n[0],
      (void**)&(a0_ptr[0]),
      k[0],
      &beta,
      (void**)&(c0_ptr[0]),
      n[0],
      num_uni);

  cuda_gemm_batched_fp32_fp32(
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      k[0],
      m[0],
      n[0],
      &alpha,
      (void**)&(b1_ptr[0]),
      n[0],
      (void**)&(a1_ptr[0]),
      n[0],
      &beta,
      (void**)&(c1_ptr[0]),
      k[0],
      num_uni);
  for (int32_t t = 0; t < T; ++t) {
    int32_t D_0 = tt_cores[t].size(1);
    int32_t tx_0 = std::min(1024, D_0);
    int32_t ty_0 = 1024 / tx_0;
    extra_fused_update_tt_cores_sgd_kernel<<<
        div_round_up(num_uni, ty_0),
        dim3(tx_0, ty_0),
        0,
        c10::cuda::getCurrentCUDAStream()>>>(
        num_uni,
        D_0,
        learning_rate,
        &(tt_idx.data_ptr<int32_t>()[t * num_uni]),
        tr_tt_cores[t].packed_accessor64<float, 2, RestrictPtrTraits>(),
        tt_cores[t].packed_accessor64<float, 2, RestrictPtrTraits>());
  }
  return;
}

__global__ void perpare_part2_ptr_kernel(
    int32_t N,
    int32_t length_d,
    int32_t length_r,
    int32_t length_l,

    float** __restrict__ a_ptr,
    float** __restrict__ b_ptr,
    float** __restrict__ c_ptr,

    const int64_t* __restrict__ tt_p_shape,

    const int64_t* __restrict__ indices,
    float* __restrict__ core2,
    float* __restrict__ core2_d,
    float* __restrict__ right,
    float* __restrict__ left,

    float** __restrict__ group_map) {
  int32_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) {
    return;
  }

  int32_t idx_start = n;
  int32_t idx_end = n + 1;

  for (int32_t i = idx_start; i < idx_end; ++i) {
    int64_t idx = __ldg(&indices[n]);

    int32_t group1 = idx / tt_p_shape[2]; // group1 = I1 * tt_p_shape[1] + I2
    int I3 = idx % tt_p_shape[2];

    a_ptr[i] = group_map[group1];
    b_ptr[i] = right + i * length_r;
    c_ptr[i] = core2_d + i * length_d;

    a_ptr[N + i] = right + i * length_r;
    b_ptr[N + i] = core2 + I3 * length_d;
    c_ptr[N + i] = left + i * length_l;
  }
}

void compute_part2_gradients(
    int32_t count, // b
    Tensor tensor_p_shapes, // (p0, p1, p2)
    Tensor tensor_q_shapes, // (q0, q1, q2)

    int32_t m,
    int32_t n,
    int32_t k,
    float alpha,
    float beta,

    at::Tensor indices, // (b, )
    at::Tensor core2_d, // (max_b, r2 * q2)
    at::Tensor core2, // (p2, r2 * q2)
    at::Tensor left, // (max_b, q0 * q1 * r2)
    at::Tensor right, // (max_b, q0 * q1 * q2)

    at::Tensor group_map) {
  auto a_ptr_tensor = at::empty({count * 2}, left.options().dtype(at::kLong));
  auto b_ptr_tensor = at::empty({count * 2}, left.options().dtype(at::kLong));
  auto c_ptr_tensor = at::empty({count * 2}, left.options().dtype(at::kLong));

  float** a_ptr = (float**)a_ptr_tensor.data_ptr<int64_t>();
  float** b_ptr = (float**)b_ptr_tensor.data_ptr<int64_t>();
  float** c_ptr = (float**)c_ptr_tensor.data_ptr<int64_t>();

  int32_t length_d = core2_d.size(1);
  int32_t length_r = right.size(1);
  int32_t length_l = left.size(1);
  int32_t length_core2 = core2.size(1);
  TORCH_CHECK(length_d == length_core2)

  int32_t threads = (count > 256 ? 256 : 32);
  int32_t blocks = (count + threads - 1) / threads;
  perpare_part2_ptr_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      count,
      length_d,
      length_r,
      length_l,
      a_ptr,
      b_ptr,
      c_ptr,
      (const int64_t*)tensor_p_shapes.data_ptr(),
      indices.data_ptr<int64_t>(),
      core2.data_ptr<float>(),
      core2_d.data_ptr<float>(),
      right.data_ptr<float>(),
      left.data_ptr<float>(),
      (float**)group_map.data_ptr());

  cuda_gemm_batched_fp32_fp32(
      CUBLAS_OP_N,
      CUBLAS_OP_T,
      n,
      k,
      m,
      &alpha,
      (void**)b_ptr,
      n,
      (void**)a_ptr,
      k,
      &beta,
      (void**)c_ptr,
      n,
      count);

  cuda_gemm_batched_fp32_fp32(
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      k,
      m,
      n,
      &alpha,
      (void**)&(b_ptr[count]),
      n,
      (void**)&(a_ptr[count]),
      n,
      &beta,
      (void**)&(c_ptr[count]),
      k,
      count);
}

__global__ void aggregate_core2_gradients_sgd_kernel(
    int32_t B,
    int32_t D,
    float learning_rate,
    const int64_t* __restrict__ tt_p_shape,
    const int64_t* __restrict__ indices,

    float* __restrict__ tt_cores_trd,
    float* __restrict__ agg_core_d) {
  int32_t n = blockIdx.x * blockDim.y + threadIdx.y;
  if (n >= B) {
    return;
  }
  auto idx = __ldg(&indices[n]) % tt_p_shape[2];
  for (int32_t d = threadIdx.x; d < D; d += blockDim.x) {
    auto delta = tt_cores_trd[n * D + d];
    atomicAdd(&(agg_core_d[idx * D + d]), delta);
  }
}

__global__ void second_aggregate_trs_gradients_kernel(
    int32_t B,
    int32_t D,
    const int64_t* __restrict__ tt_p_shape,

    const int64_t* __restrict__ indices,
    float** __restrict__ group_map,
    float* __restrict__ trs) {
  int32_t n = blockIdx.x * blockDim.y + threadIdx.y;
  if (n >= B) {
    return;
  }

  auto group = __ldg(&indices[n]) / tt_p_shape[2];
  auto store_addr = group_map[group];

  for (int32_t d = threadIdx.x; d < D; d += blockDim.x) {
    auto delta = trs[n * D + d];
    atomicAdd(&(store_addr[d]), delta);
  }
}

void second_aggregate_sgd_gradients(
    int32_t count,
    float learning_rate,

    Tensor tensor_p_shapes, // (p0, p1, p2)

    at::Tensor indices,
    at::Tensor core2_trd,
    at::Tensor d_core2,
    at::Tensor batch_trs,
    at::Tensor group_map, // pointer to agg_trs
    at::Tensor batch_agg_trs) {
  // We pass `batch_agg_trs` is just for set it zero.
  batch_agg_trs.zero_();

  int32_t length_d = d_core2.size(1);
  int32_t length_r = batch_trs.size(1);

  {
    int32_t tx = std::min(1024, length_d);
    int32_t ty = 1024 / tx;
    aggregate_core2_gradients_sgd_kernel<<<
        div_round_up(count, ty),
        dim3(tx, ty),
        0,
        c10::cuda::getCurrentCUDAStream()>>>(
        count,
        length_d,
        learning_rate,
        (const int64_t*)tensor_p_shapes.data_ptr(),
        indices.data_ptr<int64_t>(),
        core2_trd.data_ptr<float>(),
        d_core2.data_ptr<float>());
  }
  {
    int32_t tx = std::min(1024, length_r);
    int32_t ty = 1024 / tx;
    second_aggregate_trs_gradients_kernel<<<
        div_round_up(count, ty),
        dim3(tx, ty),
        0,
        c10::cuda::getCurrentCUDAStream()>>>(
        count,
        length_r,
        (const int64_t*)tensor_p_shapes.data_ptr(),
        indices.data_ptr<int64_t>(),
        (float**)group_map.data_ptr(),
        batch_trs.data_ptr<float>());
  }
}

__global__ void aggregate_core0_gradients_sgd_kernel(
    int32_t B,
    int32_t D,
    float learning_rate,
    const int64_t* __restrict__ tt_p_shape,
    const int64_t* __restrict__ indices,

    float* __restrict__ tt_cores_d,
    float* __restrict__ tt_core) {
  int32_t n = blockIdx.x * blockDim.y + threadIdx.y;
  if (n >= B) {
    return;
  }
  int32_t group = __ldg(&indices[n]) / tt_p_shape[2];
  int idx = group / tt_p_shape[1];

  for (int32_t d = threadIdx.x; d < D; d += blockDim.x) {
    auto delta = -learning_rate * tt_cores_d[n * D + d];
    atomicAdd(&(tt_core[idx * D + d]), delta);
  }
}

__global__ void aggregate_core01_gradients_kernel(
    int32_t B,
    int32_t D,
    float learning_rate,
    const int64_t* __restrict__ tt_p_shape,

    int32_t* __restrict__ tt_idx,
    float** __restrict__ tt_core_d_ptr,
    float* __restrict__ tt_core) {
  int32_t n = blockIdx.x * blockDim.y + threadIdx.y;
  if (n >= B) {
    return;
  }

  int32_t idx = __ldg(&tt_idx[n]);
  auto load_addr = tt_core_d_ptr[n];

  for (int32_t d = threadIdx.x; d < D; d += blockDim.x) {
    // auto delta = -learning_rate * __ldg(&tt_core_d[n * D + d]);
    auto delta = load_addr[d];
    atomicAdd(&(tt_core[idx * D + d]), delta);
  }
}

__global__ void perpare_core0_ptr_kernel(
    int32_t N,
    int32_t length0,
    int32_t length1,

    float** __restrict__ a_ptr,
    float** __restrict__ b_ptr,
    float** __restrict__ c_ptr,

    const int64_t* __restrict__ tt_p_shape,

    const int64_t* __restrict__ indices,
    float* __restrict__ core1,
    float* __restrict__ core0_d,
    int32_t* __restrict__ tt_idx,
    int32_t* __restrict__ group_flag,
    int32_t* __restrict__ group_idx,
    float** __restrict__ group_map) {
  int32_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) {
    return;
  }

  int32_t idx_start = n;
  int32_t idx_end = n + 1;

  for (int32_t i = idx_start; i < idx_end; ++i) {
    auto idx = __ldg(&indices[n]);
    int32_t group1 = idx / tt_p_shape[2];

    if (atomicCAS(group_flag + group1, 0, 1) == 0) {
      int cache_idx = atomicAdd(group_idx, 1);
      int I2 = group1 % tt_p_shape[1];
      // int I1 = group1 / tt_p_shape[1];
      int I1 = idx / (tt_p_shape[1] * tt_p_shape[2]);

      a_ptr[cache_idx] = group_map[group1];
      b_ptr[cache_idx] = core1 + I2 * length1;
      c_ptr[cache_idx] = core0_d + cache_idx * length0;

      tt_idx[cache_idx] = I1;
    }
  }
}

__global__ void perpare_core1_ptr_kernel(
    int32_t N,
    int32_t length0,
    int32_t length1,

    float** __restrict__ a_ptr,
    float** __restrict__ b_ptr,
    float** __restrict__ c_ptr,

    const int64_t* __restrict__ tt_p_shape,

    const int64_t* __restrict__ indices,
    float* __restrict__ core0,
    float* __restrict__ core1_d,
    int32_t* __restrict__ tt_idx,
    int32_t* __restrict__ group_flag,
    int32_t* __restrict__ group_idx,
    float** __restrict__ group_map) {
  int32_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) {
    return;
  }

  int32_t idx_start = n;
  int32_t idx_end = n + 1;

  for (int32_t i = idx_start; i < idx_end; ++i) {
    auto idx = __ldg(&indices[n]);
    int32_t group1 = idx / tt_p_shape[2];

    if (atomicCAS(group_flag + group1, 0, 1) == 0) {
      int cache_idx = atomicAdd(group_idx, 1);
      int I2 = group1 % tt_p_shape[1];
      // int I1 = group1 / tt_p_shape[1];
      int I1 = idx / (tt_p_shape[1] * tt_p_shape[2]);

      a_ptr[cache_idx] = core0 + I1 * length0;
      b_ptr[cache_idx] = group_map[group1];
      c_ptr[cache_idx] = core1_d + cache_idx * length1;

      tt_idx[cache_idx] = I2;
    }
  }
}

__global__ void perpare_core01_ptr_kernel(
    int32_t N,
    int32_t agg_count,
    int32_t length0,
    int32_t length1,

    float** __restrict__ a_ptr,
    float** __restrict__ b_ptr,
    float** __restrict__ c_ptr,

    const int64_t* __restrict__ tt_p_shape,

    const int64_t* __restrict__ indices,
    float* __restrict__ core0,
    float* __restrict__ core1,
    float* __restrict__ core0_d,
    float* __restrict__ core1_d,
    int32_t* __restrict__ tt_idx,
    int32_t* __restrict__ group_flag,
    int32_t* __restrict__ group_idx,
    float** __restrict__ group_map) {
  int32_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) {
    return;
  }

  int32_t idx_start = n;
  int32_t idx_end = n + 1;

  for (int32_t i = idx_start; i < idx_end; ++i) {
    auto idx = __ldg(&indices[n]);
    int32_t group1 = idx / tt_p_shape[2];

    if (atomicCAS(group_flag + group1, 0, 1) == 0) {
      int cache_idx = atomicAdd(group_idx, 1);
      int I2 = group1 % tt_p_shape[1];
      // int I1 = group1 / tt_p_shape[1];
      int I1 = idx / (tt_p_shape[1] * tt_p_shape[2]);

      a_ptr[cache_idx] = core0 + I1 * length0;
      b_ptr[cache_idx] = group_map[group1];
      c_ptr[cache_idx] = core1_d + cache_idx * length1; // core1

      tt_idx[cache_idx] = I2;

      a_ptr[agg_count + cache_idx] = group_map[group1];
      b_ptr[agg_count + cache_idx] = core1 + I2 * length1;
      c_ptr[agg_count + cache_idx] = core0_d + cache_idx * length0; // core0

      tt_idx[agg_count + cache_idx] = I1;
    }
  }
}

void compute_aggregate_part1_gradients(
    int32_t count,
    int32_t agg_count, // agg_b
    Tensor tensor_p_shapes, // (p0, p1, p2)
    Tensor tensor_q_shapes, // (q0, q1, q2)
    float learning_rate,

    int32_t m,
    int32_t n,
    int32_t k,
    float alpha,
    float beta,

    at::Tensor indices, // (b, )
    at::Tensor core0, // (p0, q0 * r1)
    at::Tensor core1, // (p1, r1 * q1 * r2)
    at::Tensor core0_trd, // (max_agg_b, q0 * r1)
    at::Tensor core1_trd, // (max_agg_b, r1 * q1 * r2)
    at::Tensor d_core0,
    at::Tensor d_core1,

    at::Tensor group_flags,
    at::Tensor tt_idx,
    at::Tensor gemm_count,
    at::Tensor group_map) {
  auto a_ptr_tensor = at::empty({agg_count * 2}, core0.options().dtype(at::kLong));
  auto b_ptr_tensor = at::empty({agg_count * 2}, core0.options().dtype(at::kLong));
  auto c_ptr_tensor = at::empty({agg_count * 2}, core0.options().dtype(at::kLong));

  float** a_ptr = (float**)a_ptr_tensor.data_ptr<int64_t>();
  float** b_ptr = (float**)b_ptr_tensor.data_ptr<int64_t>();
  float** c_ptr = (float**)c_ptr_tensor.data_ptr<int64_t>();

  int32_t length0 = core0.size(1);
  int32_t length0_d = core0_trd.size(1);
  int32_t length1 = core1.size(1);
  int32_t length1_d = core1_trd.size(1);
  TORCH_CHECK(length0 == length0_d && length1 == length1_d)

  {
    group_flags.zero_();
    gemm_count.zero_();

    int32_t threads = (count > 256 ? 256 : 32);
    int32_t blocks = (count + threads - 1) / threads;
    perpare_core01_ptr_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        count,
        agg_count,
        length0,
        length1,
        a_ptr,
        b_ptr,
        c_ptr,
        (const int64_t*)tensor_p_shapes.data_ptr(),
        indices.data_ptr<int64_t>(),
        core0.data_ptr<float>(),
        core1.data_ptr<float>(),
        core0_trd.data_ptr<float>(),
        core1_trd.data_ptr<float>(),
        tt_idx.data_ptr<int32_t>(),
        group_flags.data_ptr<int32_t>(),
        gemm_count.data_ptr<int32_t>(),
        (float**)group_map.data_ptr());
  }

  cuda_gemm_batched_fp32_fp32( // core1
      CUBLAS_OP_N,
      CUBLAS_OP_T,
      n,
      k,
      m,
      &alpha,
      (void**)b_ptr,
      n,
      (void**)a_ptr,
      k,
      &beta,
      (void**)c_ptr,
      n,
      agg_count);

  cuda_gemm_batched_fp32_fp32( // core0
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      k,
      m,
      n,
      &alpha,
      (void**)&(b_ptr[agg_count]),
      n,
      (void**)&(a_ptr[agg_count]),
      n,
      &beta,
      (void**)&(c_ptr[agg_count]),
      k,
      agg_count);

  {
    int32_t tx = std::min(1024, length1);
    int32_t ty = 1024 / tx;
    aggregate_core01_gradients_kernel<<<
        div_round_up(agg_count, ty),
        dim3(tx, ty),
        0,
        c10::cuda::getCurrentCUDAStream()>>>(
        agg_count,
        length1,
        learning_rate,
        (const int64_t*)tensor_p_shapes.data_ptr(),
        tt_idx.data_ptr<int32_t>(),
        c_ptr,
        d_core1.data_ptr<float>());
  }

  {
    int32_t tx = std::min(1024, length0);
    int32_t ty = 1024 / tx;
    aggregate_core01_gradients_kernel<<<
        div_round_up(agg_count, ty),
        dim3(tx, ty),
        0,
        c10::cuda::getCurrentCUDAStream()>>>(
        agg_count,
        length0,
        learning_rate,
        (const int64_t*)tensor_p_shapes.data_ptr(),
        tt_idx.data_ptr<int32_t>() + agg_count,
        c_ptr + agg_count,
        d_core0.data_ptr<float>());
  }
}

__global__ void update_tt_cores_sgd_kernel(
    int32_t B,
    int32_t D,
    float learning_rate,
    float* __restrict__ d_tt_cores,
    float* __restrict__ tt_cores) {
  int32_t b = blockIdx.x * blockDim.y + threadIdx.y;
  if (b >= B) {
    return;
  }
  for (int32_t d = threadIdx.x; d < D; d += blockDim.x) {
    tt_cores[b * D + d] -= learning_rate * d_tt_cores[b * D + d];
  }
}

void Fused_TT_3Order_backward_bag_cuda_contraction(
    int32_t num_embeddings,
    int32_t feature_dim,
    int32_t optim,
    float learning_rate,
    const std::vector<int>& tt_p_shapes, //[i1,i2,i3]
    const std::vector<int>& tt_q_shapes, //[j1,j2,j3]
    const std::vector<int>& tt_ranks, //[1,r1,r2,1]
    at::Tensor tensor_p_shapes, //[i1,i2,i3]
    at::Tensor tensor_q_shapes, //[j1,j2,j3]
    at::Tensor tensor_ranks, //[1,r1,r2,1]
    std::vector<Tensor>& tt_cores,
    at::Tensor reduced_d,
    const std::vector<std::tuple<at::Tensor, at::Tensor, at::Tensor>>& micro_batched_data,
    const std::tuple<int64_t, int64_t, std::vector<int64_t>>& micro_infos) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(reduced_d.get_device());

  int32_t T = 3;
  auto fp_options = reduced_d.options();

  bool check_inputs = true;
  if (check_inputs) {
    CHECK_INPUT(tensor_p_shapes);
    CHECK_INPUT(tensor_q_shapes);
    CHECK_INPUT(tensor_ranks);
    CHECK_INPUT(reduced_d);
    for (auto t : tt_cores) {
      CHECK_INPUT(t);
      TORCH_CHECK(t.device() == fp_options.device());
      TORCH_CHECK(t.dtype() == fp_options.dtype());
    }
    TORCH_CHECK(tensor_p_shapes.dtype() == at::kLong)
    TORCH_CHECK(tensor_q_shapes.dtype() == at::kLong)
    TORCH_CHECK(reduced_d.size(1) == feature_dim);
  }
  TORCH_CHECK(tt_p_shapes.size() == 3ull, "Only support 3-order Tensor-train.")
  TORCH_CHECK(feature_dim % 4 == 0, "feature_dim must be aligned with 4.")

  int32_t num_micros = micro_batched_data.size();
  int32_t tr_output_dim = tt_q_shapes[0] * tt_q_shapes[1] * tt_q_shapes[2];
  int32_t cache_dim = tt_q_shapes[0] * tt_q_shapes[1] * tt_ranks[2];

  auto gemm_count_t = at::empty({1}, fp_options.dtype(at::kInt));
  auto group_flags = at::empty({tt_p_shapes[0] * tt_p_shapes[1]}, fp_options.dtype(at::kInt));
  auto group_map = at::empty({tt_p_shapes[0] * tt_p_shapes[1]}, fp_options.dtype(at::kLong));

  const auto& [max_part1_gemm_cnt, max_micro_uni, part1_gemm_cnts] = micro_infos;

  // storage micro batch unique gradients
  auto micro_unique_d = at::empty({max_micro_uni, tr_output_dim}, fp_options);
  // storge micro batch recompute forward results and backward gradients
  auto batch_trs = at::empty({max_micro_uni, cache_dim}, fp_options);
  auto batch_agg_trs = at::empty({max_part1_gemm_cnt, cache_dim}, fp_options);
  // storge micro batch tt_cores gradients
  std::vector<Tensor> tr_tt_cores;
  tr_tt_cores.push_back(at::empty({max_part1_gemm_cnt, tt_cores[0].size(1)}, fp_options));
  tr_tt_cores.push_back(
      at::empty({max_part1_gemm_cnt, tt_cores[1].size(1)}, fp_options)); // Huge!!!
  tr_tt_cores.push_back(at::empty({max_micro_uni, tt_cores[2].size(1)}, fp_options));
  // storage gradient of tt_cores
  std::vector<at::Tensor> d_tt_cores;
  for (int32_t t = 0; t < T; ++t) {
    d_tt_cores.push_back(at::zeros_like(tt_cores[t]));
  }
  auto tt_idx = at::empty({max_part1_gemm_cnt * (T - 1)}, fp_options.dtype(at::kInt));

  std::vector<int32_t> m(T - 1);
  std::vector<int32_t> n(T - 1);
  std::vector<int32_t> k(T - 1);
  float alpha = 1.0;
  float beta = 0.0;
  int32_t m_ = tt_q_shapes[0];
  for (int32_t t = 0; t < T - 1; ++t) {
    m[t] = m_; // m[0]=j1 m[1]=j1*j2
    k[t] = tt_ranks[t + 1]; // k[0]=r1 k[1]=r2
    n[t] = tt_q_shapes[t + 1] * tt_ranks[t + 2]; // n[0]=j2*r2 n[1]=j3
    m_ = m_ * tt_q_shapes[t + 1];
  }

  // First backward multiple batches
  for (int32_t i = 0; i < num_micros; ++i) {
    auto& [micro_uniques, micro_back_rowidx, micro_rowidx] = micro_batched_data[i];

    auto micro_nnz = micro_rowidx.numel();
    auto micro_uni = micro_uniques.numel();

    // Fisrt contraction (batach_size -> unique_num)
    aggregate_gradients( // (unique_num, tr_output_dim)
        micro_unique_d,
        micro_nnz,
        micro_uni,
        tr_output_dim,
        feature_dim,
        micro_rowidx,
        micro_back_rowidx,
        reduced_d);

    // Recompute forward part1
    TT_3Order_forwrard_bag_cuda_no_reduce(
        micro_uniques,
        tt_p_shapes,
        tt_q_shapes,
        tt_ranks,
        tensor_p_shapes,
        tensor_q_shapes,
        tensor_ranks,
        tt_cores,
        part1_gemm_cnts[i],
        gemm_count_t,
        group_flags,
        group_map,
        batch_agg_trs,
        std::nullopt);

    // Compute part2's gradients(core2)
    compute_part2_gradients(
        micro_uni,
        tensor_p_shapes,
        tensor_q_shapes,
        m[1],
        n[1],
        k[1],
        alpha,
        beta,
        micro_uniques,
        tr_tt_cores[2],
        tt_cores[2],
        batch_trs,
        micro_unique_d,
        group_map);

    // Second contraction (unique_num -> q0 * q1), and aggregate core2's gradient
    second_aggregate_sgd_gradients(
        micro_uni,
        learning_rate,
        tensor_p_shapes,
        micro_uniques,
        tr_tt_cores[2],
        d_tt_cores[2],
        batch_trs,
        group_map,
        batch_agg_trs);

    // Compute part1's gradients(core0, core1)
    compute_aggregate_part1_gradients(
        micro_uni,
        part1_gemm_cnts[i],
        tensor_p_shapes,
        tensor_q_shapes,
        learning_rate,
        m[0],
        n[0],
        k[0],
        alpha,
        beta,
        micro_uniques,
        tt_cores[0],
        tt_cores[1],
        tr_tt_cores[0],
        tr_tt_cores[1],
        d_tt_cores[0],
        d_tt_cores[1],
        group_flags,
        tt_idx,
        gemm_count_t,
        group_map);
  }

  if (optim == OPTIM_SGD) {
    // SGD update kernel
    for (int32_t t = 0; t < T; ++t) {
      int32_t B = d_tt_cores[t].size(0);
      int32_t D = d_tt_cores[t].size(1);
      int32_t tx = std::min(1024, D);
      int32_t ty = 1024 / tx;
      update_tt_cores_sgd_kernel<<<
          div_round_up(B, ty),
          dim3(tx, ty),
          0,
          c10::cuda::getCurrentCUDAStream()>>>(
          B, D, learning_rate, d_tt_cores[t].data_ptr<float>(), tt_cores[t].data_ptr<float>());
    }
  } else {
    TORCH_WARN("Not support optimizor: ", optim);
  }

  return;
} //
