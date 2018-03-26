/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "args.h"
#include "matrix.h"
#include "qmatrix.h"
#include "vector.h"

namespace fasttext {

struct Node {
  int32_t parent;
  int32_t left;
  int32_t right;
  float weight;
  bool binary;
};

class Model {
 protected:
  std::shared_ptr<Matrix> wi_;
  std::shared_ptr<Matrix> wo_;
  std::shared_ptr<QMatrix> qwi_;
  std::shared_ptr<QMatrix> qwo_;
  std::shared_ptr<Args> args_;
  Vector hidden_;
  Vector output_;
  Vector grad_;
  int32_t hsz_;
  int32_t osz_;
  float loss_;
  int64_t nexamples_;
  std::vector<float> t_sigmoid_;
  std::vector<float> t_log_;
  // used for negative sampling:
  std::vector<int32_t> negatives_;
  size_t negpos;
  // used for hierarchical softmax:
  std::vector<std::vector<int32_t>> paths;
  std::vector<std::vector<bool>> codes;
  std::vector<Node> tree;

  static bool comparePairs(const std::pair<float, int32_t>&,
                           const std::pair<float, int32_t>&);

  int32_t getNegative(int32_t target);
  void initSigmoid();
  void initLog();

  static const int32_t NEGATIVE_TABLE_SIZE = 10000000;

 public:
  Model(std::shared_ptr<Matrix>, std::shared_ptr<Matrix>, std::shared_ptr<Args>,
        int32_t);

  float binaryLogistic(int32_t, bool, float);
  float negativeSampling(int32_t, float);
  float hierarchicalSoftmax(int32_t, float);
  float softmax(int32_t, float);

  void predict(const std::vector<int32_t>&, int32_t, float,
               std::vector<std::pair<float, int32_t>>&, Vector&, Vector&) const;
  void predict(const std::vector<int32_t>&, int32_t, float,
               std::vector<std::pair<float, int32_t>>&);
  void dfs(int32_t, float, int32_t, float,
           std::vector<std::pair<float, int32_t>>&, Vector&) const;
  void findKBest(int32_t, float, std::vector<std::pair<float, int32_t>>&,
                 Vector&, Vector&) const;
  void update(const std::vector<int32_t>&, int32_t, float);
  void computeHidden(const std::vector<int32_t>&, Vector&) const;
  void computeOutputSoftmax(Vector&, Vector&) const;
  void computeOutputSoftmax();

  void setTargetCounts(const std::vector<float>&);
  void initTableNegatives(const std::vector<float>&);
  void buildTree(const std::vector<float>&);
  float getLoss() const;
  float sigmoid(float) const;
  float log(float) const;
  float std_log(float) const;

  std::minstd_rand rng;
  bool quant_;
  void setQuantizePointer(std::shared_ptr<QMatrix>, std::shared_ptr<QMatrix>,
                          bool);
};

}  // namespace fasttext
