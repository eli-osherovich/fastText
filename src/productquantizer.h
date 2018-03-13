/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <cstring>
#include <istream>
#include <ostream>
#include <random>
#include <vector>

#include "vector.h"

namespace fasttext {

class ProductQuantizer {
 protected:
  const int32_t nbits_ = 8;
  const int32_t ksub_ = 1 << nbits_;
  const int32_t max_points_per_cluster_ = 256;
  const int32_t max_points_ = max_points_per_cluster_ * ksub_;
  const int32_t seed_ = 1234;
  const int32_t niter_ = 25;
  const float eps_ = 1e-7;

  int32_t dim_;
  int32_t nsubq_;
  int32_t dsub_;
  int32_t lastdsub_;

  std::vector<float> centroids_;

  std::minstd_rand rng;

 public:
  ProductQuantizer() {}
  ProductQuantizer(int32_t, int32_t);

  float* get_centroids(int32_t, uint8_t);
  const float* get_centroids(int32_t, uint8_t) const;

  float assign_centroid(const float*, const float*, uint8_t*, int32_t) const;
  void Estep(const float*, const float*, uint8_t*, int32_t, int32_t) const;
  void MStep(const float*, float*, const uint8_t*, int32_t, int32_t);
  void kmeans(const float*, float*, int32_t, int32_t);
  void train(int, const float*);

  float mulcode(const Vector&, const uint8_t*, int32_t, float) const;
  void addcode(Vector&, const uint8_t*, int32_t, float) const;
  void compute_code(const float*, uint8_t*) const;
  void compute_codes(const float*, uint8_t*, int32_t) const;

  void save(std::ostream&);
  void load(std::istream&);
};

}  // namespace fasttext
