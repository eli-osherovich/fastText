/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "vector.h"

#include <assert.h>

#include <cmath>
#include <iomanip>

#include <ipp.h>
#include "matrix.h"
#include "qmatrix.h"

namespace fasttext {

Vector::~Vector() { ippsFree(data_); }

Vector::Vector(std::size_t m) : size_(m) {
  data_ = static_cast<float *>(ippsMalloc_32f_L(size_));
}

void Vector::zero() { ippsZero_32f(data_, size_); }

float Vector::norm() const {
  float norm;
  ippsNorm_L2_32f(data_, size_, &norm);
  return norm;
}

void Vector::mul(float a) { ippsMulC_32f_I(a, data_, size_); }

void Vector::addVector(const Vector &source) {
  assert(size() == source.size());
  ippsAdd_32f_I(source.data(), data_, size_);
}

void Vector::addVector(const Vector &source, float a) {
  assert(size() == source.size());
  ippsAddProductC_32f(source.data(), a, data_, size_);
}

void Vector::addRow(const Matrix &A, std::size_t i) {
  assert(i < A.size(0));
  assert(size() == A.size(1));
  ippsAdd_32f_I(A.row(i), data_, size_);
}

void Vector::addRow(const Matrix &A, std::size_t i, float a) {
  assert(i < A.size(0));
  assert(size() == A.size(1));
  ippsAddProductC_32f(A.row(i), a, data_, size_);
}

void Vector::addRow(const QMatrix &A, std::size_t i) {
  assert(i >= 0);
  A.addToVector(*this, i);
}

void Vector::mul(const Matrix &A, const Vector &vec) {
  assert(A.size(0) == size());
  assert(A.size(1) == vec.size());
  for (std::size_t i = 0; i < size(); i++) {
    data_[i] = A.dotRow(vec, i);
  }
}

void Vector::mul(const QMatrix &A, const Vector &vec) {
  assert(A.getM() == size());
  assert(A.getN() == vec.size());
  for (std::size_t i = 0; i < size(); i++) {
    data_[i] = A.dotRow(vec, i);
  }
}

std::size_t Vector::argmax() {
  float max = data_[0];
  std::size_t argmax = 0;
  for (std::size_t i = 1; i < size(); i++) {
    if (data_[i] > max) {
      max = data_[i];
      argmax = i;
    }
  }
  return argmax;
}

std::ostream &operator<<(std::ostream &os, const Vector &v) {
  os << std::setprecision(5);
  for (std::size_t j = 0; j < v.size(); j++) {
    os << v[j] << ' ';
  }
  return os;
}

}  // namespace fasttext
