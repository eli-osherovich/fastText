#include "file_reader.hpp"

#include <cassert>
#include <string>

namespace fasttext {

FileReader::FileReader(const std::string& file_name,
                       std::ifstream::pos_type start,
                       std::ifstream::pos_type end)
    : start_(start), end_(end), if_(file_name) {
  assert(end_ > start_);
  reset();
}

void FileReader::reset() {
  if_.clear();
  if_.seekg(start_);
  assert(if_);
}

bool FileReader::getline(std::string* line) {
  if (if_.tellg() >= end_) {
    reset();
  }

  if (std::getline(if_, *line)) {
    return true;
  }

  if (if_.eof()) {
    reset();
    return static_cast<bool>(std::getline(if_, *line));
  }

  return false;
}
}  // namespace fasttext
