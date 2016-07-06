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
#include <chrono>
#include <thread>
using namespace std;

#include <boost/algorithm/string.hpp>
#include <sw2v.h>
#if LOCAL
#include <omp.h>
#endif

namespace sw2v {

bool SecondGreater (const pair<int, float> & a,
                 const pair<int, float> & b) {
  return a.second > b.second;
}


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
  cout << "vocab size: " << size << endl;

  int nword = freq_.size();
#if LOCAL
  model_ = vector<float>(nword * nhidden_, 0);
  for(int i = 0; i < model_.size(); i++) {
    model_[i] = (RAND01() - 0.5) / sqrt(1.0 + (float)nhidden_);
  }
#else
  int model_size = nword * nhidden_;
  vector<ps::Key> keys(nword, 0);
  vector<float> vals(nword, 0);
  for(int i = 0; i < nword; i++) {
    keys[i] = rand() % model_size;
    vals[i] = (RAND01() - 0.5) / sqrt(float(nhidden_) + 1.0);
  }
  kv_->Wait(kv_->Push(keys, vals));
  cout << "send init data ok" << endl;
#endif
  
  for(int i = 0; i < nword; i++) {
    int f = (int)pow((double)freq_[i], 0.75);
    for(int j = 0; j < f; j++) {
      negative_.push_back(i);
    }
  }

}

bool SparseWord2Vec::SkipFreqWord(int w) {
  float fw = (float)freq_[w];
  float t = 1000.0;
  float p = (fw - t) / fw;
  p -= sqrt(t / fw);
  if(RAND01() < p) return true;
  return false;
}

void SparseWord2Vec::SaveModel(const char * filename) {
  int model_size = freq_.size() * nhidden_;
  int nword = freq_.size();
  cout << "save model" << endl;
  ofstream out(filename);
  for(int w = 0; w < nword; w++) {
    if(freq_[w] < 5) continue;
    vector<ps::Key> keys(nhidden_, 0);
    for(int k = 0; k < nhidden_; k++) {
      keys[k] = w * nhidden_ + k;
    }
    vector<float> vals;
    kv_->Wait(kv_->Pull(keys, &vals));
    for(int k = 0; k < nhidden_; k++) {
      out << w << "\t" << words_[w] << "\t" << vals[k] << endl;
    }
  }
  out.close();
  cout << "save model ok" << endl;
}

float auc(vector< pair<int, float> > & label_preds) {
  sort(label_preds.begin(), label_preds.end(), SecondGreater);
  long long m = 0;
  long long n = 0;
  long long p = 0;
  int nsamples = label_preds.size();
  for(int i = 0; i < nsamples; i++) {
    pair<int, float> & e = label_preds[i];
    if(e.first == 1) {
      m += 1;
      p += (nsamples - i);
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
    vector<int> data = iter.NextWords(10000);
    cout << "data size: " << data.size() << endl;
    if(data.size() == 0) break;
    vector<Sample> batch;
    batch.reserve(data.size() * (win_size_ + 1) + 1);
#if LOCAL
#pragma omp parallel for num_threads(20)
#endif
    for(int k = 0; k < data.size(); k++) {
      int start = MAX(0, k - win_size_);
      int end = MIN(data.size() - 1, k + win_size_);
      if(SkipFreqWord(data[k])) continue;
      Sample s(1.0, data[k]);
      for(int j = start; j <= end && j < data.size(); j++) {
        if(j == k) continue;
        s.context_.push_back(data[j]);
      }
#if LOCAL
      LockAll();
#endif
      batch.push_back(s);
#if LOCAL
      UnLockAll();
#endif      
      for(int j = 0; j < win_size_; j++) {
        int k2 = rand() % negative_.size();
        if(negative_[k2] == data[k]) continue;
        if(SkipFreqWord(negative_[k2])) continue;
        Sample s2(0.0, negative_[k2]);
        s2.context_ = s.context_;
#if LOCAL
        LockAll();
#endif
        batch.push_back(s2);
#if LOCAL
        UnLockAll();
#endif
      }
    }

    int nbatch = batch.size() / batch_size_;
    vector< pair<int, float> > label_preds;
    label_preds.reserve(batch.size() + 1);

#if LOCAL
#pragma omp parallel for num_threads(20)
#endif
    for(int b = 0; b < nbatch; b++) {
      //if (b % 1000 == 0)
      //  cout << b << "\t" << nbatch << endl;
      int start = b * batch_size_;
      int end = MIN(start + batch_size_, batch.size() - 1);
      if(end <= start) continue;
      vector<Sample> sub_batch(batch.begin() + start, batch.begin() + end);
      vector< pair<int, float> > sub_label_preds = MiniBatch(sub_batch);
      for(int j = 0; j < sub_label_preds.size(); j++)
        label_preds.push_back(sub_label_preds[j]);
    }
    cout << n++ << "\t" << auc(label_preds) << endl;
    if(batch_size_ < 256) batch_size_ *= 2;
  }
  //SaveModel();
}

inline float dot(const vector<float> & f1, const vector<float> & f2) {
  int n = f1.size();
  float ret = 0.0;
  for(int i = 0; i < n; i++) {
    ret += f1[i] * f2[i];
  }
  return ret;
}

inline void AddVector(vector<float> & dst, const vector<float> & src) {
  int n = dst.size();
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
  if(batch.size() == 0) return ret;
  ret.reserve(batch.size() + 1);
  unordered_map<int, vector<float> > batch_model;
  unordered_map<int, vector<float> > grads;
  set<int> words;
  for(int i = 0; i < batch.size(); i++) {
    words.insert(batch[i].target_);
    for(int j = 0; j < batch[i].context_.size(); j++) {
      words.insert(batch[i].context_[j]);
    }
  }
  vector<int> vwords(words.begin(), words.end());
  sort(vwords.begin(), vwords.end());

#if LOCAL
  vector<int> keys;
#else
  vector<ps::Key> keys;
#endif
  for(int i = 0; i < vwords.size(); i++) {
    int w = vwords[i];
    for(int j = w*nhidden_; j < (w+1)*nhidden_; j++) {
      keys.push_back(j);
    }
  }
  
  vector<float> vals;
#if LOCAL
  vals = vector<float>(keys.size(), 0);
  for(int i = 0; i < keys.size(); i++) {
    vals[i] = model_[keys[i]];
  }
#else
  kv_->Wait(kv_->Pull(keys, &vals));
#endif
  
  CHECK_EQ(vals.size(), keys.size());

  for(int i = 0; i < vwords.size(); i++) {
    CHECK_LT(i * nhidden_, keys.size());
    int w = keys[i*nhidden_] / nhidden_;
    vector<float> sub;
    for(int j = i * nhidden_; j < i * nhidden_ + nhidden_ && j < vals.size(); j++) {
      sub.push_back(vals[j]);
    }
    CHECK_EQ(sub.size(), nhidden_);
    batch_model[w] = sub;
  }

  for(unordered_map<int, vector<float> >::const_iterator i = batch_model.begin();
      i != batch_model.end(); i++) {
    grads[i->first] = vector<float>(nhidden_, 0);
  }

  for(int i = 0; i < batch.size(); i++) {
    float pred = OneStep(batch[i].target_, batch[i].context_, batch[i].label_,
                         batch_model, grads);
    ret.push_back(pair<int, float>((int)batch[i].label_, pred));
  }

  CHECK_EQ(grads.size(), vwords.size());

  for(int i = 0; i < vwords.size(); i++) {
    CHECK_LT(i * nhidden_, keys.size());
    int w = keys[i * nhidden_] / nhidden_;
    vector<float> & sub = grads[w];
    for(int j = 0; j < nhidden_; j++) {
      CHECK_LT(i * nhidden_ + j, vals.size());
      vals[i*nhidden_ + j] = sub[j];
    }
  }

#if LOCAL
  LockAll();
  for(int i = 0; i < keys.size(); i++) {
    model_[keys[i]] += vals[i];
  }
  UnLockAll();
#else
  kv_->Wait(kv_->Push(keys, vals));
  return ret;
#endif
  
}
float SparseWord2Vec::OneStep(int w1, const vector<int> & w2s, float label,
                              const unordered_map<int, vector<float> > & batch_model,
                              unordered_map<int, vector<float> > & grads) {
  float lambda_ = 0.00001;
  float pred = 0.0;
  unordered_map<int, vector<float> >::const_iterator mi;

  mi = batch_model.find(w1);
  CHECK_NE(mi, batch_model.end());
  const vector<float> & f1 = mi->second;
  CHECK_EQ(f1.size(), nhidden_);
  vector<float> sf2(nhidden_, 0.0);

  for(int i = 0; i < w2s.size(); i++) {
    int w2 = w2s[i];
    mi = batch_model.find(w2);
    CHECK_NE(mi, batch_model.end());
    const vector<float> & f2 = mi->second;
    AddVector(sf2, f2);
    pred += dot(f1, f2);
  }

  pred = Sigmoid(pred);
  float err = (label - pred);

  unordered_map<int, vector<float> >::iterator gi = grads.find(w1);
  CHECK_NE(gi, grads.end());
  vector<float> & d1 = gi->second;

  CHECK_EQ(f1.size(), nhidden_);
  CHECK_EQ(sf2.size(), nhidden_);
  for (int i = 0; i < nhidden_; i++) {
    float v1 = f1[i];
    float v2 = sf2[i];
    d1[i] += learning_rate_ * (err * v2 - lambda_ * v1);
  }
  
  for(int i = 0; i < w2s.size(); i++) {
    int w2 = w2s[i];
    mi = batch_model.find(w2);
    CHECK_NE(mi, batch_model.end());
    
    const vector<float> & f2 = mi->second;
    CHECK_EQ(f2.size(), nhidden_);
    
    gi = grads.find(w2);
    CHECK_NE(gi, grads.end());
    vector<float> & d2 = gi->second;
    CHECK_EQ(d2.size(), nhidden_);
    for(int j = 0; j < nhidden_; j++) {
      float v1 = f1[j];
      float v2 = f2[j];      
      d2[j] += learning_rate_ * (err * v1 - lambda_ * v2);
    }
  }

  return pred;
}

}

