#pragma once

#include <fstream>
#include <string>
#include <limits>

namespace fasttext {
class FileReader {
 public:
  FileReader(const std::string& file_name, std::ifstream::pos_type start = 0,
             std::ifstream::pos_type end = std::numeric_limits<std::streamsize>::max());

  bool getline(std::string* line);

 private:
  void reset();


  std::ifstream::pos_type start_, end_;
  std::ifstream if_;

};
}  // namespace fasttext
