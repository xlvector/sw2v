#include <vector>
#include <set>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <map>
#include <string>
#include <limits>
#include <cstdlib>
#include <algorithm>
#include <exception>
using namespace std;

#include <boost/algorithm/string.hpp>
#include <sw2v.h>
#include "ps/ps.h"
namespace sw2v {

void SparseWord2Vec::LoadVocab(const char * fname) {
  ifstream in(fname);
  string line;
  getline(in, line);
  int size = atoi(line.c_str());
  words_ = vector<string>(size);
  freq_ = vector<int>(size, 0);
  while(getline(in, line)) {
    istringstream iss(line);
    string word;
    int index, freq;
    iss >> word >> index >> freq;
    words_[index] = word;
    freq_[index] = freq;
  }
  in.close();
  cout << size << endl;
  model_ = vector<float>(words_.size() * nhidden_, 0);

  for(int i = 0; i < freq_.size(); i++) {
    int f = (int)pow((double)freq_[i], 0.75);
    for(int j = 0; j < f; j++) {
      negative_.push_back(i);
    }
  }
}

inline float rand01() {
  return (float)(rand() % 100000) / 100000.0;
}

void SparseWord2Vec::InitModel() {
  float nv = sqrt(1.0 + (float)nhidden_);
  for(int i = 0; i < model_.size(); i++) {
    model_[i] = (rand01() - 0.5) / nv;
  }
}

bool SecondGreater (const pair<int, float> & a,
                 const pair<int, float> & b) {
  return a.second > b.second;
}

void SparseWord2Vec::SaveModel() {
  ofstream out("./data/text8_sw2v.model");
  vector< pair<int, float> > freqs;
  for(int i = 0; i < freq_.size(); i++) {
    if(freq_[i] < 5) continue;
    float f = (float)freq_[i];
    freqs.push_back(pair<int, float>(i, f));
  }
  sort(freqs.begin(), freqs.end(), SecondGreater);
  out << freqs.size() << " " << nhidden_ << endl;
  for(int i = 0; i < freqs.size(); i++) {
    int w = freqs[i].first;
    out << words_[w];
    int start = w * nhidden_;
    int end = start + nhidden_;
    for(int j = start; j < end; j++) {
      out << " " << model_[j];
    }
    out << endl;
  }
  out.close();
}

float auc(vector< pair<int, float> > & label_preds) {
  sort(label_preds.begin(), label_preds.end(), SecondGreater);
  long long m = 0;
  long long n = 0;
  long long p = 0;
  for(int i = 0; i < label_preds.size(); i++) {
    pair<int, float> & e = label_preds[i];
    if(e.first == 1) {
      m += 1;
      p += (label_preds.size() - i);
    }
    else n += 1;
  }
  p -= m * (m + 1);
  return (float)((double)(p) / (double)(m * n));
}


void SparseWord2Vec::Train(DataIter & iter) {
  int n = 0;
  cout << "begin training..." << endl;
  float pred = 0.0;
  while(true) {
    vector<int> data = iter.NextWords(100000);
    if(data.size() == 0) break;
    vector<Sample> batch;
    batch.reserve(data.size() * (win_size_ + 1) + 100);

    for(int k = 0; k < data.size(); k++) {
      int start = MAX(0, k - win_size_);
      int end = MIN(data.size(), k + win_size_);
      Sample s(1.0, data[k]);
      for(int j = start; j <= end; j++) {
        if(j == k) continue;
        s.context_.push_back(data[j]);
      }
      batch.push_back(s);
      
      for(int j = 0; j < win_size_; j++) {
        int k2 = rand() % negative_.size();;
        if(negative_[k2] == data[k]) continue;
        Sample s2(0.0, negative_[k2]);
        s2.context_ = s.context_;
        batch.push_back(s2);
      }
    }
    int nbatch = batch.size() / batch_size_;
    vector< pair<int, float> > label_preds;
    label_preds.reserve(batch.size() + 100);

    for(int b = 0; b < nbatch; b++) {
      if (b % 1000 == 0)
        cout << b << "\t" << nbatch << endl;
      int start = b * batch_size_;
      int end = MIN(start + batch_size_, batch.size() - 1);
      if(end <= start) continue;
      vector<Sample> sub_batch(batch.begin() + start, batch.begin() + end);
      vector< pair<int, float> > sub_label_preds = MiniBatch(sub_batch);
      for(int j = 0; j < sub_label_preds.size(); j++)
        label_preds.push_back(sub_label_preds[j]);
    }
    cout << n++ << "\t" << auc(label_preds) << endl;
    label_preds.clear();
  }
  SaveModel();
}

inline float dot(const float * f1, const float * f2, const int n) {
  float ret = 0.0;
  for(int i = 0; i < n; i++) {
    ret += f1[i] * f2[i];
  }
  return ret;
}

void AddVector(float * dst, const float * src, const int n) {
  for(int i = 0; i < n; i++) {
    dst[i] += src[i];
  }
}

inline float sign(const float x) {
  if(x < 0) return -1.0;
  else if(x > 0) return 1.0;
  else return 0.0;
}

vector< pair<int, float> > SparseWord2Vec::MiniBatch(const vector<Sample> & batch) {
  vector< pair<int, float> > ret;
  ret.reserve(batch.size() + 1);
  map<int, float*> batch_model;
  map<int, vector<float> > grads;
  set<int> words;
  for(int i = 0; i < batch.size(); i++) {
    words.insert(batch[i].target_);
    for(int j = 0; j < batch[i].context_.size(); j++) {
      words.insert(batch[i].context_[j]);
    }
  }

  vector<ps::Key> keys;
  keys.reserve(10 + words.size() * nhidden_);
  for(set<int>::iterator i = words.begin(); i != words.end(); i++) {
    int w = *i;
    for(int j = w * nhidden_; j < (w+1) * nhidden_; j++) {
      keys.push_back((ps::Key)j);
    }
  }

  vector<float> vals;
  try {
    kv_->Wait(kv_->Pull(keys, &vals));
  } catch(exception & e) {
    cout << e.what() << endl;
    return ret;
  }

  if (vals.size() != keys.size()) {
    cout << "get val error" << endl;
    return ret;
  }
  
  for(int i = 0; i < words.size(); i++) {
    int w = keys[i * nhidden_] / nhidden_;
    if((i+1) * nhidden_ > vals.size()) {
      cout << "index overflow: " << i << "\t" << nhidden_ << endl;
      return ret;
    }
    vector<float> sub(vals.begin() + i * nhidden_,
                      vals.begin() + i * nhidden_ + nhidden_);
    batch_model[w] = sub.data();
  }
  
  for(map<int, float*>::const_iterator i = batch_model.begin();
      i != batch_model.end(); i++) {
    grads[i->first] = vector<float>(nhidden_, 0);
  }
  for(int i = 0; i < batch.size(); i++) {
    float pred = OneStep(batch[i].target_, batch[i].context_, batch[i].label_,
                         batch_model, grads);
    ret.push_back(pair<int, float>((int)batch[i].label_, pred));
  }

  if(grads.size() != words.size()) {
    cout << "invalid" << endl;
    return ret;
  }

  for(int i = 0; i < words.size(); i++) {
    if (i * nhidden_ >= keys.size()) {
      cout << "index overflow" << endl;
      return ret;
    }
    int w = keys[i * nhidden_] / nhidden_;
    vector<float> & sub = grads[w];
    for(int j = 0; j < nhidden_; j++) {
      if (i * nhidden_ + j >= vals.size()) {
        cout << "index overflow" << endl;
        return ret;
      }
      vals[i*nhidden_ + j] = sub[j];
    }
  }

  kv_->Wait(kv_->Push(keys, vals));
  return ret;
}

float SparseWord2Vec::OneStep(int w1, const vector<int> & w2s, float label,
                              const map<int, float*> & batch_model,
                              map<int, vector<float> > & grads) {
  float lambda_ = 0.00001;
  float pred = 0.0;
  map<int, float*>::const_iterator mi;

  mi = batch_model.find(w1);
  const float * f1 = mi->second;
  vector<float> sf2(nhidden_, 0.0);
  
  for(int i = 0; i < w2s.size(); i++) {
    int w2 = w2s[i];
    mi = batch_model.find(w2);
    const float * f2 = mi->second;
    AddVector(sf2.data(), f2, nhidden_);
    pred += dot(f1, f2, nhidden_);
  }
  pred = Sigmoid(pred);
  float err = (label - pred);
  
  vector<float> & d1 = grads[w1];
  for (int i = 0; i < nhidden_; i++) {
    float v1 = f1[i];
    float v2 = sf2[i];
    d1[i] = learning_rate_ * (err * v2 - lambda_ * v1);
  }

  for(int i = 0; i < w2s.size(); i++) {
    int w2 = w2s[i];
    mi = batch_model.find(w2);
    const float * f2 = mi->second;

    vector<float> & d2 = grads[w2];
    for(int j = 0; j < nhidden_; j++) {
      float v1 = f1[j];
      float v2 = f2[j];      
      d2[j] = learning_rate_ * (err * v1 - lambda_ * v2);
    }
  }
  return pred;
}

}

