#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "deeplearn/layers.h"
#include "deeplearn/loss.h"
#include "deeplearn/optim.h"
#include "deeplearn/tensor.h"

namespace py = pybind11;

namespace dl {

// Helper function to convert Tensor to NumPy array
py::array_t<float> tensor_to_numpy(const Tensor &tensor) {
  auto shape = tensor.shape();
  std::vector<ssize_t> numpy_shape(shape.begin(), shape.end());

  // Create a NumPy array with the same shape
  py::array_t<float> result(numpy_shape);
  auto buf = result.request();
  float *ptr = static_cast<float *>(buf.ptr);

  // Copy data from tensor to numpy array
  const auto &data = tensor.data();
  for (size_t i = 0; i < tensor.numel(); ++i) {
    ptr[i] = data[i];
  }

  return result;
}

// Helper function to create Tensor from NumPy array
Tensor tensor_from_numpy(py::array_t<float> arr) {
  py::buffer_info buf = arr.request();

  // Convert shape
  std::vector<int> shape;
  for (ssize_t i = 0; i < buf.ndim; ++i) {
    shape.push_back(static_cast<int>(buf.shape[i]));
  }

  // Convert data
  std::vector<float> data;
  float *ptr = static_cast<float *>(buf.ptr);
  size_t size = 1;
  for (auto s : shape) {
    size *= s;
  }
  data.assign(ptr, ptr + size);

  return Tensor(shape, data);
}

} // namespace dl

PYBIND11_MODULE(deeplearn, m) {
  m.doc() = "DeepLearn Framework - A simple deep learning framework in C++";

  // ==================== Tensor Class ====================
  py::class_<dl::Tensor>(m, "Tensor")
      .def(py::init<>(), "Create an empty tensor")
      .def(py::init<const std::vector<int> &>(), py::arg("shape"),
           "Create a tensor with given shape")
      .def(py::init<const std::vector<int> &, const std::vector<float> &>(),
           py::arg("shape"), py::arg("data"),
           "Create a tensor with given shape and data")

      // Factory methods
      .def_static("from_numpy", &dl::tensor_from_numpy, py::arg("array"),
                  "Create a tensor from NumPy array")

      // Conversion methods
      .def("to_numpy", &dl::tensor_to_numpy, "Convert tensor to NumPy array")
      .def("numpy", &dl::tensor_to_numpy, "Alias for to_numpy()")
      // Properties
      .def_property_readonly("shape", &dl::Tensor::shape, "Get tensor shape")
      .def("size", &dl::Tensor::size, "Get tensor shape (alias)")
      .def("numel", &dl::Tensor::numel, "Get total number of elements")

      // Attributes
      .def_readwrite("requires_grad", &dl::Tensor::requires_grad,
                     "Whether this tensor requires gradient computation")
      .def_property(
          "grad", [](const dl::Tensor &t) -> dl::Tensor * { return t.grad; },
          [](dl::Tensor &t, dl::Tensor *g) {
            if (t.grad)
              delete t.grad;
            t.grad = (g != nullptr) ? new dl::Tensor(*g) : nullptr;
          },
          py::return_value_policy::reference, "Gradient tensor (or None)")

      // Initialization
      .def_static("from_numpy", &dl::tensor_from_numpy,
                  "Create a tensor from a NumPy array")

      // Data Access
      .def("to_numpy", &dl::tensor_to_numpy, "Convert tensor to NumPy array")
      .def("numpy", &dl::tensor_to_numpy, "Alias for to_numpy()")

      // Operations
      .def("zero_", &dl::Tensor::zero_, "Fill tensor with zeros (in-place)")
      .def("print", &dl::Tensor::print, "Print tensor")
      .def("reshape", &dl::Tensor::reshape, py::arg("new_shape"),
           "Reshape tensor")

      // Mathematical operations
      .def("add", &dl::Tensor::add, py::arg("other"), "Element-wise addition")
      .def("sub", &dl::Tensor::sub, py::arg("other"),
           "Element-wise subtraction")
      .def("mul", &dl::Tensor::mul, py::arg("other"),
           "Element-wise multiplication")
      .def("matmul", &dl::Tensor::matmul, py::arg("other"),
           "Matrix multiplication")
      .def("transpose", &dl::Tensor::transpose, "Transpose (2D only)")
      .def("sum", &dl::Tensor::sum, "Sum all elements")

      // Operators
      .def("__add__", &dl::Tensor::add)
      .def("__sub__", &dl::Tensor::sub)
      .def("__mul__", &dl::Tensor::mul)
      .def("__matmul__", &dl::Tensor::matmul)

      // Neural network operations
      .def("relu", &dl::Tensor::relu, "ReLU activation")
      .def("softmax", &dl::Tensor::softmax, py::arg("dim") = -1,
           "Softmax activation")
      .def("cross_entropy", &dl::Tensor::cross_entropy, py::arg("target"),
           "Cross entropy loss")
      .def("conv2d", &dl::Tensor::conv2d, py::arg("kernel"),
           py::arg("stride") = 1, py::arg("padding") = 0, "2D convolution")
      .def("maxpool2d", &dl::Tensor::maxpool2d, py::arg("kernel_size"),
           py::arg("stride") = 1, "2D max pooling")

      // Autograd
      .def("backward", &dl::Tensor::backward,
           "Compute gradients via backpropagation")
      .def("zero_grad", &dl::Tensor::zero_grad, "Zero out gradients")

      // String representation
      .def("__repr__", [](const dl::Tensor &t) {
        auto shape = t.shape();
        std::string shape_str = "[";
        for (size_t i = 0; i < shape.size(); ++i) {
          shape_str += std::to_string(shape[i]);
          if (i < shape.size() - 1)
            shape_str += ", ";
        }
        shape_str += "]";
        return "Tensor(shape=" + shape_str +
               ", requires_grad=" + (t.requires_grad ? "True" : "False") + ")";
      });

  // ==================== Linear Layer ====================
  py::class_<dl::Linear>(m, "Linear")
      .def(py::init<int, int>(), py::arg("in_features"),
           py::arg("out_features"), "Linear (fully connected) layer")
      .def("forward", &dl::Linear::forward, py::arg("input"), "Forward pass")
      .def("__call__", &dl::Linear::forward, py::arg("input"),
           "Forward pass (callable)")
      .def_property(
          "W", [](dl::Linear &l) { return &l.W; },
          [](dl::Linear &l, const dl::Tensor &t) { l.W = t; },
          py::return_value_policy::reference_internal, "Weight tensor")
      .def_property(
          "b", [](dl::Linear &l) { return &l.b; },
          [](dl::Linear &l, const dl::Tensor &t) { l.b = t; },
          py::return_value_policy::reference_internal, "Bias tensor")
      .def(
          "parameters",
          [](dl::Linear &layer) {
            return std::vector<dl::Tensor *>{&layer.W, &layer.b};
          },
          py::return_value_policy::reference_internal,
          "Get list of trainable parameters");

  // ==================== Conv2D Layer ====================
  py::class_<dl::Conv2D>(m, "Conv2D")
      .def(py::init<int, int, int, int, int>(), py::arg("in_channels"),
           py::arg("out_channels"), py::arg("kernel_size"),
           py::arg("stride") = 1, py::arg("padding") = 0,
           "2D Convolutional layer")
      .def("forward", &dl::Conv2D::forward, py::arg("input"), "Forward pass")
      .def("__call__", &dl::Conv2D::forward, py::arg("input"),
           "Forward pass (callable)")
      .def_property(
          "W", [](dl::Conv2D &l) { return &l.W; },
          [](dl::Conv2D &l, const dl::Tensor &t) { l.W = t; },
          py::return_value_policy::reference_internal, "Weight tensor")
      .def_property(
          "b", [](dl::Conv2D &l) { return &l.b; },
          [](dl::Conv2D &l, const dl::Tensor &t) { l.b = t; },
          py::return_value_policy::reference_internal, "Bias tensor")
      .def_readonly("stride", &dl::Conv2D::stride, "Stride")
      .def_readonly("padding", &dl::Conv2D::padding, "Padding")
      .def(
          "parameters",
          [](dl::Conv2D &layer) {
            return std::vector<dl::Tensor *>{&layer.W, &layer.b};
          },
          py::return_value_policy::reference_internal,
          "Get list of trainable parameters");

  // ==================== SGD Optimizer ====================
  py::class_<dl::SGD>(m, "SGD")
      .def(py::init<const std::vector<dl::Tensor *> &, float, float>(),
           py::arg("parameters"), py::arg("lr"), py::arg("momentum") = 0.0f,
           "SGD optimizer with optional momentum")
      .def("step", &dl::SGD::step, "Perform one optimization step")
      .def("zero_grad", &dl::SGD::zero_grad, "Zero all parameter gradients");

  // ==================== Loss Functions ====================
  py::class_<dl::CrossEntropyLoss>(m, "CrossEntropyLoss")
      .def(py::init<>(), "Cross entropy loss")
      .def(
          "__call__",
          [](dl::CrossEntropyLoss &loss, const dl::Tensor &logits,
             const dl::Tensor &targets) { return loss(logits, targets); },
          py::arg("logits"), py::arg("targets"), "Compute cross entropy loss");

  // ==================== Utility Functions ====================
  m.def("from_numpy", &dl::tensor_from_numpy, py::arg("array"),
        "Create a tensor from NumPy array (convenience function)");
}
