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
#include <memory>
#include <ostream>
#include <vector>

namespace fasttext {

class Matrix;
class QMatrix;

class Vector {
 protected:
  float *data_;
  const std::size_t size_;

 public:
  static constexpr int alignment = 64;
  explicit Vector(std::size_t);
  Vector(const Vector &) = delete;
  Vector &operator=(const Vector &) = delete;
  virtual ~Vector();

  inline float *data() { return data_; }
  inline const float *data() const { return data_; }
  inline float &operator[](std::size_t i) { return data_[i]; }
  inline const float &operator[](std::size_t i) const { return data_[i]; }

  inline std::size_t size() const { return size_; }
  void zero();
  void mul(float);
  float norm() const;
  void addVector(const Vector &source);
  void addVector(const Vector &, float);
  void addRow(const Matrix &, std::size_t);
  void addRow(const QMatrix &, std::size_t);
  void addRow(const Matrix &, std::size_t, float);
  void mul(const QMatrix &, const Vector &);
  void mul(const Matrix &, const Vector &);
  std::size_t argmax();
};

std::ostream &operator<<(std::ostream &, const Vector &);

}  // namespace fasttext
