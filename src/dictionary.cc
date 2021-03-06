/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "dictionary.h"

#include <boost/tokenizer.hpp>

#include <assert.h>

#include <algorithm>
#include <cmath>
#include <deque>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>


#define XXH_INLINE_ALL
#include "xxhash.h"

namespace fasttext {

constexpr char Dictionary::BOW[];
constexpr char Dictionary::EOW[];

Dictionary::Dictionary(std::shared_ptr<Args> args)
    : args_(args),
      word2int_(MAX_VOCAB_SIZE, -1),
      size_(0),
      nwords_(0),
      nlabels_(0),
      ntokens_(0),
      total_weight_(0),
      pruneidx_size_(-1) {}

Dictionary::Dictionary(std::shared_ptr<Args> args, std::istream& in)
    : args_(args),
      size_(0),
      nwords_(0),
      nlabels_(0),
      ntokens_(0),
      total_weight_(0),
      pruneidx_size_(-1) {
  load(in);
}

int32_t Dictionary::find(const std::string& w) const {
  return find(w, hash(w));
}

int32_t Dictionary::find(const std::string& w, uint32_t h) const {
  int32_t word2intsize = word2int_.size();
  int32_t id = h % word2intsize;
  while (word2int_[id] != -1 && words_[word2int_[id]].word != w) {
    id = (id + 1 < word2intsize) ? id + 1 : 0;
  }
  return id;
}

void Dictionary::add(const std::string& w, float weight) {
  int32_t h = find(w);
  ntokens_++;
  total_weight_ += weight;
  if (word2int_[h] == -1) {
    entry e;
    e.word = w;
    e.weight = weight;
    e.type = getType(w);
    words_.push_back(e);
    word2int_[h] = size_++;
  } else {
    words_[word2int_[h]].weight += weight;
  }
}

int32_t Dictionary::nwords() const { return nwords_; }

int32_t Dictionary::nlabels() const { return nlabels_; }

int64_t Dictionary::ntokens() const { return ntokens_; }

const std::vector<int32_t>& Dictionary::getSubwords(int32_t i) const {
  assert(i >= 0);
  assert(i < nwords_);
  return words_[i].subwords;
}

const std::vector<int32_t> Dictionary::getSubwords(
    const std::string& word) const {
  int32_t i = getId(word);
  if (i >= 0) {
    return getSubwords(i);
  }

  return computeSubwords(word, args_->minn, args_->maxn, BOW, EOW);
}

void Dictionary::getSubwords(const std::string& word,
                             std::vector<int32_t>& ngrams,
                             std::vector<std::string>& substrings) const {
  int32_t i = getId(word);
  ngrams.clear();
  substrings.clear();
  if (i >= 0) {
    ngrams.push_back(i);
    substrings.push_back(words_[i].word);
  }
  computeSubwords(BOW + word + EOW, ngrams, substrings);
}

bool Dictionary::discard(int32_t id, float rand, float boost) const {
  assert(id >= 0);
  assert(id < nwords_);
  if (args_->model == model_name::sup) return false;
  return rand > pdiscard_[id] * boost;
}

int32_t Dictionary::getId(const std::string& w, uint32_t h) const {
  int32_t id = find(w, h);
  return word2int_[id];
}

int32_t Dictionary::getId(const std::string& w) const {
  int32_t h = find(w);
  return word2int_[h];
}

entry_type Dictionary::getType(int32_t id) const {
  assert(id >= 0);
  assert(id < size_);
  return words_[id].type;
}

entry_type Dictionary::getType(const std::string& w) const {
  return (w.find(args_->label) == 0) ? entry_type::label : entry_type::word;
}

std::string Dictionary::getWord(int32_t id) const {
  assert(id >= 0);
  assert(id < size_);
  return words_[id].word;
}
uint32_t Dictionary::hash(const std::string& str) const {
  return XXH32(str.data(), str.length(), /*seed*/ 0);
}

void Dictionary::computeSubwords(const std::string& word,
                                 std::vector<int32_t>& ngrams,
                                 std::vector<std::string>& substrings) const {
  for (size_t i = 0; i < word.size(); i++) {
    std::string ngram;
    if ((word[i] & 0xC0) == 0x80) continue;
    for (size_t j = i, n = 1; j < word.size() && n <= args_->maxn; n++) {
      ngram.push_back(word[j++]);
      while (j < word.size() && (word[j] & 0xC0) == 0x80) {
        ngram.push_back(word[j++]);
      }
      if (n >= args_->minn && !(n == 1 && (i == 0 || j == word.size()))) {
        int32_t h = hash(ngram) % args_->bucket;
        ngrams.push_back(nwords_ + h);
        substrings.push_back(ngram);
      }
    }
  }
}

std::vector<int32_t> Dictionary::computeSubwords(const std::string& word,
                                                 unsigned int min_len,
                                                 unsigned int max_len,
                                                 const std::string& bow,
                                                 const std::string& eow) const {
  std::vector<int32_t> res;
  if (max_len > word.size()) {
    max_len = word.size();
  }
  for (std::string::size_type len = min_len; len <= max_len; ++len) {
    for (std::string::size_type start = 0; start <= word.size() - len;
         ++start) {
      std::string subword = word.substr(start, len);
      if (start == 0) {
        subword = bow + subword;
      }
      if (start + len == word.size()) {
        subword += eow;
      }
      int32_t h = hash(subword) % args_->bucket;
      res.push_back(h);
    }
  }
  return res;
}

void Dictionary::initNgrams() {
  for (size_t i = 0; i < size_; i++) {
    words_[i].subwords =
        computeSubwords(words_[i].word, args_->minn, args_->maxn, BOW, EOW);
    words_[i].subwords.push_back(i);
  }
}

void Dictionary::readFromFile(std::istream& in) {
  std::string cur_line;
  int64_t minThreshold = 1;
  float weight = 1.0f;

  while (std::getline(in, cur_line)) {
    // Special treatment for the first column that may be 'weight'.
    std::size_t p = 0;
    if (args_->has_weight) {
      weight = std::stof(cur_line, &p);
    }
    // The rest of the line is a sequence of words.
    boost::tokenizer<> tok(cur_line.cbegin() + p, cur_line.cend());
    for (const std::string& word : tok) {
      add(word, weight);
      if (ntokens_ % 1000000 == 0 && args_->verbose > 1) {
        std::cerr << "\rRead " << ntokens_ / 1000000 << "M words" << std::flush;
      }
      if (size_ > 0.75 * MAX_VOCAB_SIZE) {
        minThreshold++;
        threshold(minThreshold, minThreshold);
      }
    }
  }
  threshold(args_->minCount, args_->minCountLabel);
  initTableDiscard();
  initNgrams();
  if (args_->verbose > 0) {
    std::cerr << "\rRead " << ntokens_ / 1000000 << "M words" << std::endl;
    std::cerr << "Number of words:  " << nwords_ << std::endl;
    std::cerr << "Number of labels: " << nlabels_ << std::endl;
  }
  if (size_ == 0) {
    throw std::invalid_argument(
        "Empty vocabulary. Try a smaller -minCount value.");
  }
}

void Dictionary::threshold(int64_t t, int64_t tl) {
  sort(words_.begin(), words_.end(), [](const entry& e1, const entry& e2) {
    if (e1.type != e2.type) return e1.type < e2.type;
    return e1.weight > e2.weight;
  });
  words_.erase(
      remove_if(words_.begin(), words_.end(),
                [&](const entry& e) {
                  return (e.type == entry_type::word && e.weight < t) ||
                         (e.type == entry_type::label && e.weight < tl);
                }),
      words_.end());
  words_.shrink_to_fit();
  size_ = 0;
  nwords_ = 0;
  nlabels_ = 0;
  std::fill(word2int_.begin(), word2int_.end(), -1);
  for (auto it = words_.begin(); it != words_.end(); ++it) {
    int32_t h = find(it->word);
    word2int_[h] = size_++;
    if (it->type == entry_type::word) nwords_++;
    if (it->type == entry_type::label) nlabels_++;
  }
}

void Dictionary::initTableDiscard() {
  pdiscard_.resize(size_);
  for (size_t i = 0; i < size_; i++) {
    float f = words_[i].weight / total_weight_;
    pdiscard_[i] = std::sqrt(args_->t / f) + args_->t / f;
  }
}

std::vector<float> Dictionary::getCounts(entry_type type) const {
  std::vector<float> counts;
  for (auto& w : words_) {
    if (w.type == type) counts.push_back(w.weight);
  }
  return counts;
}

void Dictionary::addWordNgrams(std::vector<int32_t>& line,
                               const std::vector<int32_t>& hashes,
                               int32_t n) const {
  for (int32_t i = 0; i < hashes.size(); i++) {
    uint64_t h = hashes[i];
    for (int32_t j = i + 1; j < hashes.size() && j < i + n; j++) {
      h = h * 116049371 + hashes[j];
      pushHash(line, h % args_->bucket);
    }
  }
}

void Dictionary::addSubwords(std::vector<int32_t>& line,
                             const std::string& token, int32_t wid) const {
  if (wid < 0) {  // out of vocab
    line = computeSubwords(token, args_->minn, args_->maxn, BOW, EOW);
  } else {
    if (args_->maxn <= 0) {  // in vocab w/o subwords
      line.push_back(wid);
    } else {  // in vocab w/ subwords
      const std::vector<int32_t>& ngrams = getSubwords(wid);
      line.insert(line.end(), ngrams.cbegin(), ngrams.cend());
    }
  }
}

void Dictionary::reset(std::istream& in) const {
  if (in.eof()) {
    in.clear();
    in.seekg(std::streampos(0));
  }
}

int32_t Dictionary::convertLine(const std::string& line, std::minstd_rand& rng,
                                std::vector<int32_t>* words,
                                float* weight) const {
  std::uniform_real_distribution<> uniform(0, 1);

  words->clear();

  // Special treatment for the first column that may be 'weight'.
  std::size_t p = 0;
  if (args_->has_weight) {
    *weight = std::stof(line, &p);
  } else {
    *weight = 1.0f;
  }

  // The rest of the line is a sequence of words.
  boost::tokenizer<> tok(line.cbegin() + p, line.cend());
  int32_t ntokens = 0;
  for (const std::string& token : tok) {
    int32_t wid = getId(token);
    if (wid < 0) continue;

    ++ntokens;
    if (getType(wid) == entry_type::word &&
        !discard(wid, uniform(rng) /*, cur_weight_*/)) {
      words->push_back(wid);
    }
  }
  return ntokens;
}

int32_t Dictionary::getLine(std::istream& in, std::vector<int32_t>& words,
                            std::vector<int32_t>& labels) const {
  std::vector<int32_t> word_hashes;
  std::string token;
  int32_t ntokens = 0;

  // reset(in);
  // words.clear();
  // labels.clear();
  // while (readWord(in, token)) {
  //   uint32_t h = hash(token);
  //   int32_t wid = getId(token, h);
  //   entry_type type = wid < 0 ? getType(token) : getType(wid);

  //   ntokens++;
  //   if (type == entry_type::word) {
  //     addSubwords(words, token, wid);
  //     word_hashes.push_back(h);
  //   } else if (type == entry_type::label && wid >= 0) {
  //     labels.push_back(wid - nwords_);
  //   }
  //   if (token == EOS) break;
  // }
  // addWordNgrams(words, word_hashes, args_->wordNgrams);
  return ntokens;
}

void Dictionary::pushHash(std::vector<int32_t>& hashes, int32_t id) const {
  if (pruneidx_size_ == 0 || id < 0) return;
  if (pruneidx_size_ > 0) {
    if (pruneidx_.count(id)) {
      id = pruneidx_.at(id);
    } else {
      return;
    }
  }
  hashes.push_back(nwords_ + id);
}

std::string Dictionary::getLabel(int32_t lid) const {
  if (lid < 0 || lid >= nlabels_) {
    throw std::invalid_argument("Label id is out of range [0, " +
                                std::to_string(nlabels_) + "]");
  }
  return words_[lid + nwords_].word;
}

void Dictionary::save(std::ostream& out) const {
  out.write((char*)&size_, sizeof(size_));
  out.write((char*)&nwords_, sizeof(nwords_));
  out.write((char*)&nlabels_, sizeof(nlabels_));
  out.write((char*)&ntokens_, sizeof(ntokens_));
  out.write((char*)&total_weight_, sizeof(total_weight_));
  out.write((char*)&pruneidx_size_, sizeof(pruneidx_size_));
  for (int32_t i = 0; i < size_; i++) {
    entry e = words_[i];
    out.write(e.word.data(), e.word.size() * sizeof(*e.word.data()));
    out.put(0);
    out.write((char*)&(e.weight), sizeof(e.weight));
    out.write((char*)&(e.type), sizeof(e.type));
  }
  for (const auto pair : pruneidx_) {
    out.write((char*)&(pair.first), sizeof(pair.first));
    out.write((char*)&(pair.second), sizeof(pair.second));
  }
}

void Dictionary::load(std::istream& in) {
  words_.clear();
  in.read((char*)&size_, sizeof(size_));
  in.read((char*)&nwords_, sizeof(nwords_));
  in.read((char*)&nlabels_, sizeof(nlabels_));
  in.read((char*)&ntokens_, sizeof(ntokens_));
  in.read((char*)&total_weight_, sizeof(total_weight_));
  in.read((char*)&pruneidx_size_, sizeof(pruneidx_size_));
  for (int32_t i = 0; i < size_; i++) {
    char c;
    entry e;
    while ((c = in.get()) != 0) {
      e.word.push_back(c);
    }
    in.read((char*)&e.weight, sizeof(e.weight));
    in.read((char*)&e.type, sizeof(e.type));
    words_.push_back(e);
  }
  pruneidx_.clear();
  for (int32_t i = 0; i < pruneidx_size_; i++) {
    int32_t first;
    int32_t second;
    in.read((char*)&first, sizeof(first));
    in.read((char*)&second, sizeof(second));
    pruneidx_[first] = second;
  }
  initTableDiscard();
  initNgrams();

  int32_t word2intsize = std::ceil(size_ / 0.7);
  word2int_.assign(word2intsize, -1);
  for (int32_t i = 0; i < size_; i++) {
    word2int_[find(words_[i].word)] = i;
  }
}

void Dictionary::prune(std::vector<int32_t>& idx) {
  std::vector<int32_t> words, ngrams;
  for (auto it = idx.cbegin(); it != idx.cend(); ++it) {
    if (*it < nwords_) {
      words.push_back(*it);
    } else {
      ngrams.push_back(*it);
    }
  }
  std::sort(words.begin(), words.end());
  idx = words;

  if (ngrams.size() != 0) {
    int32_t j = 0;
    for (const auto ngram : ngrams) {
      pruneidx_[ngram - nwords_] = j;
      j++;
    }
    idx.insert(idx.end(), ngrams.begin(), ngrams.end());
  }
  pruneidx_size_ = pruneidx_.size();

  std::fill(word2int_.begin(), word2int_.end(), -1);

  int32_t j = 0;
  for (int32_t i = 0; i < words_.size(); i++) {
    if (getType(i) == entry_type::label ||
        (j < words.size() && words[j] == i)) {
      words_[j] = words_[i];
      word2int_[find(words_[j].word)] = j;
      j++;
    }
  }
  nwords_ = words.size();
  size_ = nwords_ + nlabels_;
  words_.erase(words_.begin() + size_, words_.end());
  initNgrams();
}

void Dictionary::dump(std::ostream& out) const {
  out << words_.size() << std::endl;
  for (auto it : words_) {
    std::string entryType = "word";
    if (it.type == entry_type::label) {
      entryType = "label";
    }
    out << it.word << " " << it.weight << " " << entryType << std::endl;
  }
}

}  // namespace fasttext
