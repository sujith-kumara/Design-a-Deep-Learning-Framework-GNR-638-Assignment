#pragma once

#include <cstddef>
#include <vector>

namespace dl {

class Tensor {
public:
  Tensor();
  explicit Tensor(const std::vector<size_t> &shape);
  ~Tensor();

  // Placeholder for tensor operations
};

} // namespace dl
