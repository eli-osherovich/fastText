/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "matrix.h"
#include <iostream>

#include <cmath>
#include <exception>
#include <random>
#include <stdexcept>

#include <mkl.h>

#include "utils.h"
#include "vector.h"

namespace fasttext {
Matrix::~Matrix() { mkl_free(data_); }

Matrix::Matrix() : Matrix(0, 0) {}

Matrix::Matrix(std::size_t m, std::size_t n) : m_(m), n_(n) {
  stride_ =
      std::ceil(static_cast<float>(n_ * sizeof(float)) / Vector::alignment) *
      Vector::alignment / sizeof(float);

  data_ = static_cast<float*>(
      mkl_malloc(sizeof(float) * m_ * stride_, Vector::alignment));
}

void Matrix::zero() { std::fill(data_, data_ + m_ * stride_, 0.0f); }

void Matrix::uniform(float a) {
  std::minstd_rand rng(1);
  std::uniform_real_distribution<> uniform(-a, a);
  for (std::size_t i = 0; i < (m_ * n_); i++) {
    data_[i] = uniform(rng);
  }
}

float Matrix::dotRow(const Vector& vec, std::size_t i) const {
  assert(i < m_);
  assert(vec.size() == n_);
  float d = cblas_sdot(n_, data_ + i * stride_, 1, vec.data(), 1);
  if (std::isnan(d)) {
    throw std::runtime_error("Encountered NaN.");
  }
  return d;
}

void Matrix::addRow(const Vector& vec, std::size_t i, float a) {
  assert(i < m_);
  assert(vec.size() == n_);
  cblas_saxpy(n_, a, vec.data(), 1, data_ + i * stride_, 1);
}

// void Matrix::multiplyRow(const Vector& nums, std::size_t ib, int64_t ie) {
//   if (ie == -1) {
//     ie = m_;
//   }
//   assert(ie <= nums.size());
//   for (int64_t i = ib; i < ie; ++i) {
//     float n = nums[i - ib];
//     if (n != 0) {
//       for (auto j = 0; j < n_; j++) {
//         at(i, j) *= n;
//       }
//     }
//   }
// }

void Matrix::divideRow(const Vector& denoms, std::size_t ib, int64_t ie) {
  if (ie == -1) {
    ie = m_;
  }
  assert(ie <= denoms.size());
  for (auto i = ib; i < ie; i++) {
    float n = denoms[i - ib];
    if (n != 0) {
      for (auto j = 0; j < n_; j++) {
        at(i, j) /= n;
      }
    }
  }
}

float Matrix::l2NormRow(std::size_t i) const {
  float norm = cblas_snrm2(n_, data_ + i * stride_, 1);

  if (std::isnan(norm)) {
    throw std::runtime_error("Encountered NaN.");
  }
  return std::sqrt(norm);
}

void Matrix::l2NormRow(Vector& norms) const {
  assert(norms.size() == m_);
  for (auto i = 0; i < m_; i++) {
    norms[i] = l2NormRow(i);
  }
}

void Matrix::save(std::ostream& out) {
  out.write((char*)&m_, sizeof(std::size_t));
  out.write((char*)&n_, sizeof(std::size_t));
  out.write((char*)data_, m_ * stride_ * sizeof(float));
}

void Matrix::load(std::istream& in) {
  in.read((char*)&m_, sizeof(std::size_t));
  in.read((char*)&n_, sizeof(std::size_t));
  data_ = static_cast<float*>(
      mkl_malloc(sizeof(float) * m_ * stride_, Vector::alignment));
  in.read((char*)data_, m_ * stride_ * sizeof(float));
}

void Matrix::dump(std::ostream& out) const {
  out << m_ << " " << n_ << std::endl;
  for (std::size_t i = 0; i < m_; i++) {
    for (std::size_t j = 0; j < n_; j++) {
      if (j > 0) {
        out << " ";
      }
      out << at(i, j);
    }
    out << std::endl;
  }
};

}  // namespace fasttext
