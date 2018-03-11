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

#include "matrix.h"
#include "qmatrix.h"
#include <mkl.h>

namespace fasttext {

Vector::~Vector() { mkl_free(data_); }

Vector::Vector(std::size_t m) : size_(m) {
  data_ = static_cast<float *>(mkl_malloc(sizeof(float) * size_, alignment));
}

void Vector::zero() { std::fill(data_, data_ + size_, 0.0f); }

real Vector::norm() const { return cblas_snrm2(size_, data_, 1); }

void Vector::mul(real a) { cblas_sscal(size_, a, data_, 1); }

void Vector::addVector(const Vector &source) {
  assert(size() == source.size());
  vsAdd(size_, data_, source.data(), data_);
}

void Vector::addVector(const Vector &source, real a) {
  assert(size() == source.size());
  cblas_saxpy(size_, a, source.data(), 1, data_, 1);
}

void Vector::addRow(const Matrix &A, std::size_t i) {
  assert(i >= 0);
  assert(i < A.size(0));
  assert(size() == A.size(1));
  for (std::size_t j = 0; j < A.size(1); j++) {
    data_[j] += A.at(i, j);
  }
}

void Vector::addRow(const Matrix &A, std::size_t i, real a) {
  assert(i >= 0);
  assert(i < A.size(0));
  assert(size() == A.size(1));
  for (std::size_t j = 0; j < A.size(1); j++) {
    data_[j] += a * A.at(i, j);
  }
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
  real max = data_[0];
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

} // namespace fasttext
