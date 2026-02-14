#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "deeplearn/layers.h"
#include "deeplearn/loss.h"
#include "deeplearn/optim.h"
#include "deeplearn/tensor.h"

namespace py = pybind11;

namespace dl {

// Helper function to create Tensor from flat Python list and shape
Tensor tensor_from_list(const std::vector<float> &data,
                        const std::vector<int> &shape) {
  return Tensor(shape, data);
}

// Helper function to convert Tensor data to flat Python list
std::vector<float> tensor_to_list(const Tensor &tensor) {
  return tensor.data();
}

} // namespace dl

PYBIND11_MODULE(deeplearn, m) {
  m.doc() = "DeepLearn Framework - A simple deep learning framework in C++";

  // ==================== Enums ====================
  py::enum_<dl::Device>(m, "Device")
      .value("CPU", dl::Device::CPU)
      .value("GPU", dl::Device::GPU)
      .export_values();

  // ==================== Tensor Class ====================
  py::class_<dl::Tensor>(m, "Tensor")
      .def(py::init<>(), "Create an empty tensor")
      .def(py::init<const std::vector<int> &, dl::Device>(), py::arg("shape"),
           py::arg("device") = dl::Device::CPU)
      .def(py::init<const std::vector<int> &, const std::vector<float> &,
                    dl::Device>(),
           py::arg("shape"), py::arg("data"),
           py::arg("device") = dl::Device::CPU)

      // Device management
      .def("to", &dl::Tensor::to, py::arg("device"), "Move tensor to device")
      .def("device", &dl::Tensor::device, "Get tensor device")

      // Factory methods
      .def_static("from_list", &dl::tensor_from_list, py::arg("data"),
                  py::arg("shape"), "Create a tensor from flat list and shape")

      // Conversion methods
      .def("tolist", &dl::tensor_to_list, "Convert tensor data to flat list")

      // Properties
      .def_property_readonly("shape", &dl::Tensor::shape, "Get tensor shape")
      .def("size", &dl::Tensor::size, "Get tensor shape (alias)")
      .def("numel", &dl::Tensor::numel, "Get total number of elements")

      // Attributes
      .def_readwrite("requires_grad", &dl::Tensor::requires_grad)
      .def_property(
          "grad", [](const dl::Tensor &t) -> dl::Tensor * { return t.grad; },
          [](dl::Tensor &t, dl::Tensor *g) {
            if (t.grad)
              delete t.grad;
            t.grad = (g != nullptr) ? new dl::Tensor(*g) : nullptr;
          },
          py::return_value_policy::reference, "Gradient tensor")

      // Operations
      .def("zero_", &dl::Tensor::zero_)
      .def("print", &dl::Tensor::print)
      .def("reshape", &dl::Tensor::reshape, py::arg("new_shape"))

      // Math
      .def("add", &dl::Tensor::add, py::arg("other"), py::keep_alive<0, 1>(),
           py::keep_alive<0, 2>())
      .def("sub", &dl::Tensor::sub, py::arg("other"), py::keep_alive<0, 1>(),
           py::keep_alive<0, 2>())
      .def("mul", &dl::Tensor::mul, py::arg("other"), py::keep_alive<0, 1>(),
           py::keep_alive<0, 2>())
      .def("matmul", &dl::Tensor::matmul, py::arg("other"),
           py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
      .def("transpose", &dl::Tensor::transpose, py::keep_alive<0, 1>())
      .def("sum", &dl::Tensor::sum, py::keep_alive<0, 1>())

      // Operators
      .def("__add__", &dl::Tensor::add, py::keep_alive<0, 1>(),
           py::keep_alive<0, 2>())
      .def("__sub__", &dl::Tensor::sub, py::keep_alive<0, 1>(),
           py::keep_alive<0, 2>())
      .def("__mul__", &dl::Tensor::mul, py::keep_alive<0, 1>(),
           py::keep_alive<0, 2>())
      .def("__matmul__", &dl::Tensor::matmul, py::keep_alive<0, 1>(),
           py::keep_alive<0, 2>())

      // NN Ops
      .def("relu", &dl::Tensor::relu, py::keep_alive<0, 1>())
      .def("softmax", &dl::Tensor::softmax, py::arg("dim") = -1,
           py::keep_alive<0, 1>())
      .def("cross_entropy", &dl::Tensor::cross_entropy, py::arg("target"),
           py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
      .def("conv2d", &dl::Tensor::conv2d, py::arg("kernel"),
           py::arg("stride") = 1, py::arg("padding") = 0,
           py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
      .def("maxpool2d", &dl::Tensor::maxpool2d, py::arg("kernel_size"),
           py::arg("stride") = 1, py::keep_alive<0, 1>())

      // Autograd
      .def("backward", &dl::Tensor::backward)
      .def("zero_grad", &dl::Tensor::zero_grad)

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
           py::arg("out_features"))
      .def("forward", &dl::Linear::forward, py::arg("input"),
           py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
      .def("__call__", &dl::Linear::forward, py::arg("input"),
           py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
      .def_property(
          "W", [](dl::Linear &l) { return &l.W; },
          [](dl::Linear &l, const dl::Tensor &t) { l.W = t; },
          py::return_value_policy::reference_internal)
      .def_property(
          "b", [](dl::Linear &l) { return &l.b; },
          [](dl::Linear &l, const dl::Tensor &t) { l.b = t; },
          py::return_value_policy::reference_internal)
      .def(
          "parameters",
          [](dl::Linear &layer) {
            return std::vector<dl::Tensor *>{&layer.W, &layer.b};
          },
          py::return_value_policy::reference_internal);

  // ==================== Conv2D Layer ====================
  py::class_<dl::Conv2D>(m, "Conv2D")
      .def(py::init<int, int, int, int, int>(), py::arg("in_channels"),
           py::arg("out_channels"), py::arg("kernel_size"),
           py::arg("stride") = 1, py::arg("padding") = 0)
      .def("forward", &dl::Conv2D::forward, py::arg("input"),
           py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
      .def("__call__", &dl::Conv2D::forward, py::arg("input"),
           py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
      .def_property(
          "W", [](dl::Conv2D &l) { return &l.W; },
          [](dl::Conv2D &l, const dl::Tensor &t) { l.W = t; },
          py::return_value_policy::reference_internal)
      .def_property(
          "b", [](dl::Conv2D &l) { return &l.b; },
          [](dl::Conv2D &l, const dl::Tensor &t) { l.b = t; },
          py::return_value_policy::reference_internal)
      .def_readonly("stride", &dl::Conv2D::stride)
      .def_readonly("padding", &dl::Conv2D::padding)
      .def(
          "parameters",
          [](dl::Conv2D &layer) {
            return std::vector<dl::Tensor *>{&layer.W, &layer.b};
          },
          py::return_value_policy::reference_internal);

  // ==================== SGD Optimizer ====================
  py::class_<dl::SGD>(m, "SGD")
      .def(py::init<const std::vector<dl::Tensor *> &, float, float>(),
           py::arg("parameters"), py::arg("lr"), py::arg("momentum") = 0.0f)
      .def("step", &dl::SGD::step)
      .def("zero_grad", &dl::SGD::zero_grad);

  // ==================== Loss Functions ====================
  py::class_<dl::CrossEntropyLoss>(m, "CrossEntropyLoss")
      .def(py::init<>())
      .def(
          "__call__",
          [](dl::CrossEntropyLoss &loss, const dl::Tensor &logits,
             const dl::Tensor &targets) { return loss(logits, targets); },
          py::arg("logits"), py::arg("targets"), py::keep_alive<0, 1>(),
          py::keep_alive<0, 2>(), py::keep_alive<0, 3>());
}
