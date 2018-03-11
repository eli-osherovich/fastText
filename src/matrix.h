/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <cstdint>
#include <istream>
#include <ostream>
#include <vector>

#include <assert.h>

namespace fasttext {


class Vector;

class Matrix {
 protected:
  float* data_;
  const std::size_t m_;
  const std::size_t n_;
  std::size_t stride_;

 public:
  Matrix();
  explicit Matrix(std::size_t, std::size_t);
  Matrix(const Matrix&) = default;
  Matrix& operator=(const Matrix&) = delete;
  virtual ~Matrix();

  inline float* data() { return data_; }
  inline const float* data() const { return data_; }
  inline const float& at(int64_t i, int64_t j) const {
    return data_[i * stride_ + j];
  };

  inline float& at(int64_t i, int64_t j) { return data_[i * stride_ + j]; };

  inline int64_t size(int64_t dim) const {
    assert(dim == 0 || dim == 1);
    if (dim == 0) {
      return m_;
    }
    return n_;
  }
  inline int64_t rows() const { return m_; }
  inline int64_t cols() const { return n_; }
  void zero();
  void uniform(float);
  float dotRow(const Vector&, std::size_t) const;
  void addRow(const Vector&, std::size_t, float);

  void multiplyRow(const Vector& nums, std::size_t ib = 0, int64_t ie = -1);
  void divideRow(const Vector& denoms, std::size_t ib = 0, int64_t ie = -1);

  float l2NormRow(std::size_t i) const;
  void l2NormRow(Vector& norms) const;

  void save(std::ostream&);
  void load(std::istream&);

  void dump(std::ostream&) const;
};
}  // namespace fasttext
