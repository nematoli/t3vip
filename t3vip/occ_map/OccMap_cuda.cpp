#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

int OccMap_cuda_forward(const torch::Tensor *points,
                        torch::Tensor *indexmap,
                        torch::Tensor *output,
                        float fy, float fx, float cy, float cx);

int OccMap_cuda_backward(const torch::Tensor *input,
                         const torch::Tensor *indexmap,
                         torch::Tensor *gradInput,
                         const torch::Tensor *gradOutput,
                         float fy, float fx, float cy, float cx);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) TORCH_CHECK(x->type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x->is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

int OccMap_forward(torch::Tensor *points,
                   torch::Tensor *indexmap,
                   torch::Tensor *output,
                   float fy, float fx,
                   float cy, float cx)
 {

  CHECK_INPUT(points);
  CHECK_INPUT(indexmap);
  CHECK_INPUT(output);


  OccMap_cuda_forward(points, indexmap, output, fy, fx, cy, cx);

  return 1;
}



int OccMap_backward(torch::Tensor *input,
                    torch::Tensor *indexMap,
                    torch::Tensor *gradInput,
                    torch::Tensor *gradOutput,
                    float fy, float fx,
                    float cy, float cx) {

  CHECK_INPUT(input);
  CHECK_INPUT(indexMap);
  // CHECK_INPUT(gradInput);
  CHECK_INPUT(gradOutput);

  OccMap_cuda_backward(input, indexMap, gradInput, gradOutput, fy, fx, cy, cx);

  return 1;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &OccMap_forward, "OccMap forward (CUDA)");
  m.def("backward", &OccMap_backward, "OccMap backward (CUDA)");
}