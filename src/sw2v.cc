#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <map>
#include <string>
#include <limits>
#include <deque>
#include <cstdlib>
using namespace std;

#include <boost/algorithm/string.hpp>
#include <sw2v.h>

namespace sw2v {

void SparseWord2Vec::LoadData(const char * fname) {
  ifstream in(fname, ios::binary | ios::in);
  int val = 0;
  int max_val = 0;
  while(in.read((char*)&val, 4)) {
    data_.push_back(val);
    max_val = MAX(max_val, val);
  }
  cout << "doc count: " << data_.size() << endl;
  cout << "word count: " << max_val + 1 << endl;
  model_ = vector< vector<float> >(max_val + 1);
}

void SparseWord2Vec::InitModel() {
  float nv = sqrt((float)nhidden_);
  for(int i = 0; i < model_.size(); i++) {
    vector<float> vec(nhidden_);
    for(int j = 0; j < nhidden_; j++) {
      float r = (float)(rand() % 10000) / 10000.0;
      vec[j] = (r - 0.5) / nv;
    }
    model_[i] = vec;
  }
}

void SparseWord2Vec::Train() {
  float total_err = 0.0;
  float total = 0.0;
  
  for(int i = 0; i < data_.size() * 10; i++) {
    int k = rand() % data_.size();
    int step = (rand() % 11) - 5;
    int k2 = MIN(MAX(0, k + step), data_.size() - 1);

    if (data_[k] == data_[k2]) continue;
    total_err += abs(OneStep(data_[k], data_[k2], 1.0));
    total += 1.0;
    
    k2 = rand() % data_.size();
    bool ok = true;
    int start = MAX(0, i - 5);
    int end = MIN(data_.size(), i + 5);
    for(int j = start; j <= end; j++) {
      if(data_[k2] == data_[j]) {
        ok = false;
        break;
      }
    }
    if (!ok) continue;
    total_err += abs(OneStep(data_[k], data_[k2], 0.0));
    total += 1.0;

    if ((1+i) % 10000 == 0){
      cout << (1+i) / 10000 << "\t" << total_err / total << endl;
      total_err = 0.0;
      total = 0.0;
    }
  }
}

inline float dot(const vector<float> & f1,
                 const vector<float> & f2) {
  float ret = 0.0;
  for(int i = 0; i < f1.size(); i++) {
    ret += f1[i] * f2[i];
  }
  return 1.0 / (1.0 + exp(-1.0 * ret));
}

float SparseWord2Vec::OneStep(int w1, int w2, float label) {
  float learning_rate_ = 0.1;
  float lambda_ = 0.01;
  vector<float> & f1 = model_[w1];
  vector<float> & f2 = model_[w2];

  float pred = dot(f1, f2);
  float err = (label - pred);

  for(int i = 0; i < f1.size(); i++) {
    float v1 = f1[i];
    float v2 = f2[i];

    f1[i] = v1 + learning_rate_ * (err * v2 - lambda_ * v1);
    f2[i] = v2 + learning_rate_ * (err * v1 - lambda_ * v2);
  }
  return err;
}

}

