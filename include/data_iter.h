#ifndef DATA_ITER_H_
#define DATA_ITER_H_

#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
using namespace std;

class DataIter {
private:
  ifstream fin_;
  istream_iterator<int> in_, eof_;
public:
  DataIter(const char * name) {
    fin_.open(name, ios::in);
    in_ = istream_iterator<int>(fin_);
  }

  vector<int> NextWords(const int nword) {
    vector<int> words;
    for(int i = 0; i < nword; i++) {
      if(in_ == eof_) break;
      int word = *in_;
      ++in_;
      words.push_back(word);
    }
    return words;
  }

  ~DataIter() {
    fin_.close();
  }
};

#endif  // DATA_ITER_H_
