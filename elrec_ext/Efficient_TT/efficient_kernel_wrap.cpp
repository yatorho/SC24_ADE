#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace at;

void init_cuda(
    int32_t device_id,
    const std::vector<int>& tt_p_shape,
    const std::vector<int>& tt_q_shape,
    const std::vector<int>& tt_ranks, //[1,r1,r2,1]
    int32_t batch_size,
    int32_t feature_dim
);

void check_init(
    int32_t device_id,
    const std::vector<int>& tt_p_shape,
    const std::vector<int>& tt_q_shape,
    const std::vector<int>& tt_ranks, //[1,r1,r2,1]
    int32_t batch_size,
    int32_t feature_dim,
    at::Tensor tensor_group_map,
    at::Tensor tensor_group_flag,
    at::Tensor tensor_group_idx,
    at::Tensor tensor_cache
);

Tensor Efficient_TT_forward_cuda(
    int32_t batch_size,
    int32_t table_length,
    int32_t feature_dim,
    const Tensor index,
    const std::vector<int>& tt_p_shapes,
    const std::vector<int>& tt_q_shapes,
    const std::vector<int>& tt_ranks,
    const Tensor tenser_p_shapes,
    const Tensor tenser_q_shapes,
    const Tensor tenser_ranks,
    const std::vector<Tensor>& tt_cores,
    at::Tensor tensor_group_map,
    at::Tensor tensor_group_flag,
    at::Tensor tensor_group_idx,
    at::Tensor tensor_cache,
    at::Tensor tensor_batch_cnt
);


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
    Tensor sorted_key
    );

Tensor Efficient_TT_forward_bag_cuda(
    int32_t batch_cnt,
    int32_t table_length,
    int32_t feature_dim,
    Tensor index,
    Tensor offsets,
    Tensor rowidx,
    const std::vector<int>& tt_p_shape, //[i1,i2,i3]
    const std::vector<int>& tt_q_shape, //[j1,j2,j3]
    const std::vector<int>& tt_ranks, //[1,r1,r2,1]
    Tensor tensor_p_shape, //[i1,i2,i3]
    Tensor tensor_q_shape, //[j1,j2,j3]
    Tensor tensor_ranks, //[1,r1,r2,1]
    const std::vector<Tensor>& tt_cores,
    at::Tensor tensor_group_map,
    at::Tensor tensor_group_flag,
    at::Tensor tensor_group_idx,
    at::Tensor tensor_cache,
    at::Tensor tensor_batch_cnt);

at::Tensor compute_rowidx_cuda(
    Tensor indices,
    Tensor offsets
);


void Fused_Extra_Efficient_TT_bag_backward_sgd_cuda(
    int32_t batch_cnt,
    int32_t table_length,
    int32_t feature_dim,
    float learning_rate,
    const Tensor index,
    const Tensor rowidx,
    const std::vector<int>& tt_p_shapes, //[i1,i2,i3]
    const std::vector<int>& tt_q_shapes, //[j1,j2,j3]
    const std::vector<int>& tt_ranks, //[1,r1,r2,1]
    const Tensor tensor_p_shapes, //[i1,i2,i3]
    const Tensor tensor_q_shapes, //[j1,j2,j3]
    const Tensor tensor_ranks, //[1,r1,r2,1]
    Tensor d_reduced_output,
    std::vector<Tensor>& tt_cores,
    Tensor sorted_idx,
    Tensor sorted_key
);

PYBIND11_MODULE(elrec_kernel, m) {
  m.def("Eff_TT_forward", &Efficient_TT_forward_cuda, "tt_forward()");
  m.def("Eff_TT_bag_forward", &Efficient_TT_forward_bag_cuda, "tt_bag_forward()");
  m.def("compute_rowidx", &compute_rowidx_cuda, "compute_rowidx()");
  m.def("Eff_TT_backward", &Efficient_TT_backward_sgd_cuda, "tt_slice_backward()");
  m.def("Fused_Eff_TT_backward", &Fused_Efficient_TT_backward_sgd_cuda, "tt_slice_backward()");
  m.def("Fused_Extra_Eff_TT_backward", &Fused_Extra_Efficient_TT_backward_sgd_cuda, "tt_slice_backward()");
  m.def("Fused_Extra_Eff_TT_bag_backward", &Fused_Extra_Efficient_TT_bag_backward_sgd_cuda, "tt_slice_backward()");
  m.def("init_cuda", &init_cuda, "init_cuda()");
  m.def("check_init", &check_init, "chech_init()");
}