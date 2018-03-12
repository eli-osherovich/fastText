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

#include <ipp.h>

#include "utils.h"
#include "vector.h"

namespace fasttext {
Matrix::~Matrix() { ippsFree(data_); }

Matrix::Matrix() : Matrix(0, 0) {}

Matrix::Matrix(std::size_t m, std::size_t n) : m_(m), n_(n) {
  stride_ = std::ceil(static_cast<float>(n_ * sizeof(float)) / 64) * 64 /
            sizeof(float);
  data_ = ippsMalloc_32f_L(m_ * stride_);
}

void Matrix::zero() { ippsZero_32f(data_, m_ * stride_); }

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
  float d;
  ippsDotProd_32f(data_ + i * stride_, vec.data(), n_, &d);
  if (std::isnan(d)) {
    throw std::runtime_error("Encountered NaN.");
  }
  return d;
}

void Matrix::addRow(const Vector& vec, std::size_t i) {
  ippsAdd_32f_I(vec.data(), data_ + i * stride_, n_);
}

void Matrix::addRow(const Vector& vec, std::size_t i, float a) {
  assert(i < m_);
  assert(vec.size() == n_);
  ippsAddProductC_32f(vec.data(), a, data_ + i * stride_, n_);
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
  float norm;
  ippsNorm_L2_32f(data_, n_, &norm);

  if (std::isnan(norm)) {
    throw std::runtime_error("Encountered NaN.");
  }
  return norm;
}

void Matrix::l2NormRow(Vector& norms) const {
  assert(norms.size() == m_);
  for (auto i = 0; i < m_; i++) {
    norms[i] = l2NormRow(i);
  }
}

void Matrix::save(std::ostream& out) {
  out.write((char*)&m_, sizeof(m_));
  out.write((char*)&n_, sizeof(n_));
  out.write((char*)data_, m_ * stride_ * sizeof(*data_));
}

void Matrix::load(std::istream& in) {

  in.read((char*)&m_, sizeof(m_));
  in.read((char*)&n_, sizeof(n_));
  stride_ = std::ceil(static_cast<float>(n_ * sizeof(float)) / 64) * 64 /
    sizeof(float);
  data_ = ippsMalloc_32f_L(m_ * stride_);
  in.read((char*)data_, m_ * stride_ * sizeof(*data_));
}

void Matrix::dump(std::ostream& out) const {
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
