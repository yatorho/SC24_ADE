#include <ATen/ATen.h>
#include <torch/extension.h>

using namespace at;

std::vector<at::Tensor> BuildReusingMatrices(
    const std::vector<at::Tensor>& tt_indices,
    const std::vector<int>& tt_p_shapes);

void Efficient_TT_backward_sgd_cuda(
    int32_t batch_size,
    int32_t table_length,
    int32_t feature_dim,
    float learning_rate,

    const Tensor indices,
    const std::vector<int32_t>& tt_p_shapes,
    const std::vector<int32_t>& tt_q_shapes,
    const std::vector<int32_t>& tt_ranks,
    const Tensor tensor_p_shape,
    const Tensor tensor_q_shape,
    const Tensor tensor_ranks,
    Tensor d_output,
    std::vector<Tensor>& tt_cores);

void Fused_Efficient_TT_backward_sgd_cuda(
    int32_t batch_size,
    int32_t table_length,
    int32_t feature_dim,
    float learning_rate,

    const Tensor indices,
    const std::vector<int32_t>& tt_p_shapes,
    const std::vector<int32_t>& tt_q_shapes,
    const std::vector<int32_t>& tt_ranks,
    const Tensor tensor_p_shape,
    const Tensor tensor_q_shape,
    const Tensor tensor_ranks,
    Tensor d_output,
    std::vector<Tensor>& tt_cores);

void Fused_Extra_Efficient_TT_backward_sgd_cuda(
    int32_t batch_size,
    int32_t table_length,
    int32_t feature_dim,
    float learning_rate,

    const Tensor indices,
    const std::vector<int32_t>& tt_p_shapes,
    const std::vector<int32_t>& tt_q_shapes,
    const std::vector<int32_t>& tt_ranks,
    const Tensor tensor_p_shape,
    const Tensor tensor_q_shape,
    const Tensor tensor_ranks,
    Tensor d_output,
    std::vector<Tensor>& tt_cores,
    Tensor sorted_idx,
    Tensor sorted_key);

at::Tensor TT_3Order_forward_bag_cuda(
    int32_t num_embeddings,
    int32_t feature_dim,
    int32_t batch_size,
    at::Tensor rowidx,
    at::Tensor unique_values,
    at::Tensor unique_keys, // unique_values -> indices
    at::Tensor unique_counts,
    at::Tensor back_map,
    at::Tensor bce_idx, // [bce_fron_idx + bce_back_idx] #
                        // unique_values[bce_map] = bce_idx
    at::Tensor bce_map, // bce_idx -> unique_values
    int32_t bce_front_num,
    const std::vector<int>& tt_p_shape, //[i1,i2,i3]
    const std::vector<int>& tt_q_shape, //[j1,j2,j3]
    const std::vector<int>& tt_ranks, //[1,r1,r2,1]
    Tensor tensor_p_shape, //[i1,i2,i3]
    Tensor tensor_q_shape, //[j1,j2,j3]
    Tensor tensor_ranks, //[1,r1,r2,1]
    const std::vector<Tensor>& tt_cores);

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
    const std::vector<Tensor>& tt_cores);

at::Tensor compute_rowidx_cuda(Tensor indices, Tensor offsets);

at::Tensor aggregate_gradients(
    int32_t nnz, // batch count
    int32_t num_uni,
    int32_t tr_output_dim,
    int32_t feature_dim,
    at::Tensor rowidx,
    at::Tensor unique_keys,
    at::Tensor reduced_d);

at::Tensor aggregate_gradients_v1(
    int32_t N, // batch count
    int32_t B, // batch size
    int32_t feature_dim,
    at::Tensor d_reduced_output,
    at::Tensor rowidx,
    at::Tensor sorted_idx,
    at::Tensor sorted_key);

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
    at::Tensor reduced_d);

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
    const std::tuple<int64_t, int64_t, std::vector<int64_t>> &micro_infos
    );

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> enhance_unique(at::Tensor x);

PYBIND11_MODULE(_C, m) {
  m.def("build_reusing_matrices", &BuildReusingMatrices, "tt_forward()");
  m.def("tt_3d_forward_bag", &TT_3Order_forward_bag_cuda, "tt_bag_forward()");
  m.def(
      "tt_3d_forward_bag_with_uidx_batched",
      &TT_3Order_forward_bag_cuda_with_Uidx_batched,
      "tt_bag_forward()");
  m.def("compute_rowidx", &compute_rowidx_cuda, "compute_rowidx()");
  m.def("aggregate_gradients", &aggregate_gradients, "aggregate_gradients()");
  m.def("aggregate_gradients_v1", &aggregate_gradients_v1, "aggregate_gradients_v1()");
  m.def("fused_tt_3d_backward_bag", &Fused_TT_3Order_backward_bag_cuda, "tt backward()");
  m.def(
      "fused_tt_3d_backward_bag_contraction",
      Fused_TT_3Order_backward_bag_cuda_contraction,
      "ecorec optimized backward");
  m.def("enhance_unique", &enhance_unique, "enhance_unique()");
}