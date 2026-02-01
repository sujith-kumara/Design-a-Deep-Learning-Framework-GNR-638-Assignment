#include "deeplearn/tensor.h"
#include <iomanip>
#include <iostream> // Added for std::cout, std::endl
#include <numeric>

namespace dl {

Tensor::Tensor() : grad(nullptr), requires_grad(false), op_type("") {}

Tensor::Tensor(const std::vector<int> &shape)
    : _shape(shape), grad(nullptr), requires_grad(false), op_type("") {
  size_t total_size = numel();
  _data.assign(total_size, 0.0f);
}

Tensor::~Tensor() {
  // Note: grad ownership should be handled carefully later in autograd
}

size_t Tensor::numel() const {
  if (_shape.empty())
    return 0;
  size_t total = 1;
  for (int s : _shape)
    total *= s;
  return total;
}

void Tensor::reshape(const std::vector<int> &new_shape) {
  size_t new_total = 1;
  for (int s : new_shape)
    new_total *= s;

  if (new_total != numel()) {
    throw std::runtime_error(
        "Reshape failed: total number of elements must remain the same");
  }
  _shape = new_shape;
}

void Tensor::zero_() { std::fill(_data.begin(), _data.end(), 0.0f); }

void Tensor::print() const {
  std::cout << "Tensor(shape=[";
  for (size_t i = 0; i < _shape.size(); ++i) {
    std::cout << _shape[i] << (i == _shape.size() - 1 ? "" : ", ");
  }
  std::cout << "], data=[";

  size_t n = numel();
  size_t limit = std::min(n, (size_t)100);
  for (size_t i = 0; i < limit; ++i) {
    std::cout << std::fixed << std::setprecision(2) << _data[i]
              << (i == n - 1 ? "" : ", ");
  }
  if (n > 100)
    std::cout << "...";
  std::cout << "])" << std::endl;
}

size_t Tensor::get_flat_index(const std::vector<int> &indices) const {
  size_t flat_index = 0;
  size_t stride = 1;
  for (int i = (int)_shape.size() - 1; i >= 0; --i) {
    flat_index += indices[i] * stride;
    stride *= _shape[i];
  }
  return flat_index;
}

float &Tensor::operator()(const std::vector<int> &indices) {
  return _data[get_flat_index(indices)];
}

float Tensor::operator()(const std::vector<int> &indices) const {
  return _data[get_flat_index(indices)];
}

Tensor Tensor::add(const Tensor &other) const {
  if (_shape != other._shape)
    throw std::runtime_error("Shape mismatch for add");
  Tensor res(_shape);
  for (size_t i = 0; i < _data.size(); ++i)
    res._data[i] = _data[i] + other._data[i];

  if (this->requires_grad || other.requires_grad) {
    res.requires_grad = true;
    res.op_type = "add";
    res.parents = {(Tensor *)this, (Tensor *)&other};
  }
  return res;
}

Tensor Tensor::sub(const Tensor &other) const {
  if (_shape != other._shape)
    throw std::runtime_error("Shape mismatch for sub");
  Tensor res(_shape);
  for (size_t i = 0; i < _data.size(); ++i)
    res._data[i] = _data[i] - other._data[i];

  if (this->requires_grad || other.requires_grad) {
    res.requires_grad = true;
    res.op_type = "sub";
    res.parents = {(Tensor *)this, (Tensor *)&other};
  }
  return res;
}

Tensor Tensor::mul(const Tensor &other) const {
  if (_shape != other._shape)
    throw std::runtime_error("Shape mismatch for mul");
  Tensor res(_shape);
  for (size_t i = 0; i < _data.size(); ++i)
    res._data[i] = _data[i] * other._data[i];

  if (this->requires_grad || other.requires_grad) {
    res.requires_grad = true;
    res.op_type = "mul";
    res.parents = {(Tensor *)this, (Tensor *)&other};
  }
  return res;
}

Tensor Tensor::matmul(const Tensor &other) const {
  if (_shape.size() != 2 || other._shape.size() != 2)
    throw std::runtime_error("Matmul only for 2D");
  if (_shape[1] != other._shape[0])
    throw std::runtime_error("Incompatible shapes for matmul");

  int M = _shape[0];
  int K = _shape[1];
  int N = other._shape[1];
  Tensor res({M, N});

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += (*this)({i, k}) * other({k, j});
      }
      res({i, j}) = sum;
    }
  }

  if (this->requires_grad || other.requires_grad) {
    res.requires_grad = true;
    res.op_type = "matmul";
    res.parents = {(Tensor *)this, (Tensor *)&other};
  }
  return res;
}

Tensor Tensor::conv2d(const Tensor &kernel, int stride, int padding) const {
  if (_shape.size() != 4 || kernel._shape.size() != 4)
    throw std::runtime_error("Conv2d expects 4D input and kernel");

  int B = _shape[0];
  int C = _shape[1];
  int H = _shape[2];
  int W = _shape[3];

  int Cout = kernel._shape[0];
  int Cin = kernel._shape[1];
  int kH = kernel._shape[2];
  int kW = kernel._shape[3];

  if (C != Cin)
    throw std::runtime_error("Channel mismatch in conv2d");

  int Hout = (H + 2 * padding - kH) / stride + 1;
  int Wout = (W + 2 * padding - kW) / stride + 1;

  Tensor res({B, Cout, Hout, Wout});

  for (int b = 0; b < B; ++b) {
    for (int co = 0; co < Cout; ++co) {
      for (int h = 0; h < Hout; ++h) {
        for (int w = 0; w < Wout; ++w) {
          float val = 0;
          for (int ci = 0; ci < Cin; ++ci) {
            for (int kh = 0; kh < kH; ++kh) {
              for (int kw = 0; kw < kW; ++kw) {
                int ih = h * stride + kh - padding;
                int iw = w * stride + kw - padding;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                  val += (*this)({b, ci, ih, iw}) * kernel({co, ci, kh, kw});
                }
              }
            }
          }
          res({b, co, h, w}) = val;
        }
      }
    }
  }

  if (this->requires_grad || kernel.requires_grad) {
    res.requires_grad = true;
    res.op_type = "conv2d";
    res.parents = {(Tensor *)this, (Tensor *)&kernel};
    res.stride = stride;
    res.padding = padding;
    res.kernel_size = kH; // Assuming square kernel
  }
  return res;
}

Tensor Tensor::maxpool2d(int kernel_size, int stride) const {
  if (_shape.size() != 4)
    throw std::runtime_error("Maxpool2d expects 4D input");

  int B = _shape[0];
  int C = _shape[1];
  int H = _shape[2];
  int W = _shape[3];

  int Hout = (H - kernel_size) / stride + 1;
  int Wout = (W - kernel_size) / stride + 1;

  Tensor res({B, C, Hout, Wout});

  for (int b = 0; b < B; ++b) {
    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < Hout; ++h) {
        for (int w = 0; w < Wout; ++w) {
          float max_val = -1e30f; // Very small value
          for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
              int ih = h * stride + kh;
              int iw = w * stride + kw;
              max_val = std::max(max_val, (*this)({b, c, ih, iw}));
            }
          }
          res({b, c, h, w}) = max_val;
        }
      }
    }
  }

  if (this->requires_grad) {
    res.requires_grad = true;
    res.op_type = "maxpool2d";
    res.parents = {(Tensor *)this};
    res.kernel_size = kernel_size;
    res.stride = stride;
  }
  return res;
}

void Tensor::zero_grad() {
  if (grad) {
    grad->zero_();
  }
  for (auto p : parents)
    p->zero_grad();
}

void Tensor::backward() {
  if (requires_grad && !grad) {
    if (numel() == 1) {
      grad = new Tensor({1});
      grad->_data[0] = 1.0f;
    } else {
      throw std::runtime_error(
          "Backward can only be called on a scalar or if grad is already set.");
    }
  }

  if (!grad || op_type == "")
    return;

  if (op_type == "add") {
    for (auto p : parents) {
      if (p->requires_grad) {
        if (!p->grad)
          p->grad = new Tensor(p->_shape);
        for (size_t i = 0; i < grad->_data.size(); ++i)
          p->grad->_data[i] += grad->_data[i];
      }
    }
  } else if (op_type == "sub") {
    if (parents[0]->requires_grad) {
      if (!parents[0]->grad)
        parents[0]->grad = new Tensor(parents[0]->_shape);
      for (size_t i = 0; i < grad->_data.size(); ++i)
        parents[0]->grad->_data[i] += grad->_data[i];
    }
    if (parents[1]->requires_grad) {
      if (!parents[1]->grad)
        parents[1]->grad = new Tensor(parents[1]->_shape);
      for (size_t i = 0; i < grad->_data.size(); ++i)
        parents[1]->grad->_data[i] -= grad->_data[i];
    }
  } else if (op_type == "mul") {
    if (parents[0]->requires_grad) {
      if (!parents[0]->grad)
        parents[0]->grad = new Tensor(parents[0]->_shape);
      for (size_t i = 0; i < grad->_data.size(); ++i)
        parents[0]->grad->_data[i] += grad->_data[i] * parents[1]->_data[i];
    }
    if (parents[1]->requires_grad) {
      if (!parents[1]->grad)
        parents[1]->grad = new Tensor(parents[1]->_shape);
      for (size_t i = 0; i < grad->_data.size(); ++i)
        parents[1]->grad->_data[i] += grad->_data[i] * parents[0]->_data[i];
    }
  } else if (op_type == "matmul") {
    // Y = A @ B -> dA = dY @ B^T, dB = A^T @ dY
    Tensor &A = *parents[0];
    Tensor &B = *parents[1];
    if (A.requires_grad) {
      if (!A.grad)
        A.grad = new Tensor(A._shape);
      int M = A._shape[0], K = A._shape[1], N = B._shape[1];
      for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k)
          for (int j = 0; j < N; ++j)
            A.grad->operator()({i, k}) += grad->operator()({i, j}) * B({k, j});
    }
    if (B.requires_grad) {
      if (!B.grad)
        B.grad = new Tensor(B._shape);
      int M = A._shape[0], K = A._shape[1], N = B._shape[1];
      for (int k = 0; k < K; ++k)
        for (int j = 0; j < N; ++j)
          for (int i = 0; i < M; ++i)
            B.grad->operator()({k, j}) += A({i, k}) * grad->operator()({i, j});
    }
  } else if (op_type == "conv2d") {
    Tensor &X = *parents[0];
    Tensor &K = *parents[1];
    int B = X._shape[0], C = X._shape[1], H = X._shape[2], W = X._shape[3];
    int Cout = K._shape[0], Cin = K._shape[1], kH = K._shape[2],
        kW = K._shape[3];
    int Hout = (H + 2 * padding - kH) / stride + 1;
    int Wout = (W + 2 * padding - kW) / stride + 1;

    if (X.requires_grad) {
      if (!X.grad)
        X.grad = new Tensor(X._shape);
      for (int b = 0; b < B; ++b)
        for (int co = 0; co < Cout; ++co)
          for (int h = 0; h < Hout; ++h)
            for (int w = 0; w < Wout; ++w)
              for (int ci = 0; ci < Cin; ++ci)
                for (int kh = 0; kh < kH; ++kh)
                  for (int kw = 0; kw < kW; ++kw) {
                    int ih = h * stride + kh - padding;
                    int iw = w * stride + kw - padding;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                      X.grad->operator()({b, ci, ih, iw}) +=
                          grad->operator()({b, co, h, w}) * K({co, ci, kh, kw});
                  }
    }
    if (K.requires_grad) {
      if (!K.grad)
        K.grad = new Tensor(K._shape);
      for (int b = 0; b < B; ++b)
        for (int co = 0; co < Cout; ++co)
          for (int h = 0; h < Hout; ++h)
            for (int w = 0; w < Wout; ++w)
              for (int ci = 0; ci < Cin; ++ci)
                for (int kh = 0; kh < kH; ++kh)
                  for (int kw = 0; kw < kW; ++kw) {
                    int ih = h * stride + kh - padding;
                    int iw = w * stride + kw - padding;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                      K.grad->operator()({co, ci, kh, kw}) +=
                          grad->operator()({b, co, h, w}) * X({b, ci, ih, iw});
                  }
    }
  } else if (op_type == "maxpool2d") {
    Tensor &X = *parents[0];
    int B = X._shape[0], C = X._shape[1], H = X._shape[2], W = X._shape[3];
    int Hout = (H - kernel_size) / stride + 1;
    int Wout = (W - kernel_size) / stride + 1;

    if (X.requires_grad) {
      if (!X.grad)
        X.grad = new Tensor(X._shape);
      for (int b = 0; b < B; ++b)
        for (int c = 0; c < C; ++c)
          for (int h = 0; h < Hout; ++h)
            for (int w = 0; w < Wout; ++w) {
              float max_val = -1e30f;
              int max_h = -1, max_w = -1;
              for (int kh = 0; kh < kernel_size; ++kh)
                for (int kw = 0; kw < kernel_size; ++kw) {
                  int ih = h * stride + kh;
                  int iw = w * stride + kw;
                  if (X({b, c, ih, iw}) > max_val) {
                    max_val = X({b, c, ih, iw});
                    max_h = ih;
                    max_w = iw;
                  }
                }
              if (max_h != -1)
                X.grad->operator()({b, c, max_h, max_w}) +=
                    grad->operator()({b, c, h, w});
            }
    }
  }

  // Recursively call backward (naive approach)
  for (auto p : parents)
    p->backward();
}

} // namespace dl
