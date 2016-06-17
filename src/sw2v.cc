#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <map>
#include <string>
#include <limits>
#include <cstdlib>
#include <algorithm>
using namespace std;

#include <boost/algorithm/string.hpp>
#include <sw2v.h>

namespace sw2v {

void SparseWord2Vec::LoadData(const char * fname) {
  ifstream in(fname);
  string line;
  while(getline(in, line)) {
    vector<string> words;
    boost::split(words, line, boost::is_any_of("\t "));
    for(int i = 0; i < words.size(); i++) {
      string word = words[i];
      map<string, int>::iterator k = word_index_.find(word);
      if (k == word_index_.end()) {
        word_index_[word] = words_.size();
        words_.push_back(word);
        model_.push_back(vector<float>(nhidden_, 0));
      }
      data_.push_back(word_index_[word]);
    }
  }
  cout << "doc count: " << data_.size() << endl;
  cout << "word count: " << word_index_.size() << endl;

  for(int i = 0; i < data_.size(); i++) {
    if(data_[i] >= model_.size()) {
      cout << data_[i] << "\t" << model_.size() << endl;
      exit(1);
    }
  }
}

void SparseWord2Vec::InitModel() {
  float nv = sqrt((float)nhidden_);
  for(int i = 0; i < model_.size(); i++) {
    vector<float> & vec = model_[i];
    for(int j = 0; j < nhidden_; j++) {
      float r = (float)(rand() % 10000) / 10000.0;
      vec[j] = (r - 0.5) / nv;
    }
  }
}

bool SecondGreater (const pair<string, float> & a,
                 const pair<string, float> & b) {
  return a.second > b.second;
}

void SparseWord2Vec::SaveModel() {
  ofstream out("./data/text8.model");
  for(int i = 0; i < model_.size(); i++) {
    out << words_[i];
    for(int j = 0; j < model_[i].size(); j++) {
      out << "\t" << model_[i][j];
    }
    out << endl;
  }
  out.close();
}

void SparseWord2Vec::PrintHiddens() {
  vector< vector< pair<string, float> > > hiddens(nhidden_);
  for(int i = 0; i < model_.size(); i++) {
    for(int j = 0; j < model_[i].size(); j++) {
      hiddens[j].push_back(make_pair<string, float>(words_[i], model_[i][j]));
    }
  }

  for(int h = 0; h < nhidden_; h++) {
    sort(hiddens[h].begin(), hiddens[h].end(), SecondGreater);
    cout << "hidden: " << h << endl;
    for(int i = 0; i < 32 && i < hiddens[h].size(); i++) {
      cout << "\t" << hiddens[h][i].first << "\t"
           << hiddens[h][i].second << endl;
    }
  }
}

void SparseWord2Vec::Train() {
  float total = 0.0;
  float total_err = 0.0;

  long long n = 0;
  cout << "begin training..." << endl;
  for(int i = 0; i < 100; i++) {
#pragma omp parallel for
    for(int k = 0; k < data_.size(); k++) {
      int start = MAX(0, k - 5);
      int end = MIN(data_.size(), k + 5);
      vector<int> win;
      for(int j = start; j <= end; j++) {
        if(j == k) continue;
        win.push_back(data_[j]);
      }
      
      total_err += abs(OneStep(data_[k], win, 1.0));
      total += 1.0;

      for(int j = 0; j < 10; j++) {
        int k2 = rand() % data_.size();
        total_err += abs(OneStep(data_[k2], win, 0.0));
        total += 1.0;
      }
      if (++n % 100000 == 0){
        cout << n / 10000 << "\t" << total_err / total << endl;
        total_err = 0.0;
        total = 0.0;
      }
    }
    cout << "finish " << i << endl;
    PrintHiddens();
    SaveModel();
    learning_rate_ *= 0.9;
  }
}

inline float dot(const vector<float> & f1,
                 const vector<float> & f2) {
  float ret = 0.0;
  for(int i = 0; i < f1.size(); i++) {
    ret += f1[i] * f2[i];
  }
  return ret;
}

float SparseWord2Vec::OneStep(int w1, vector<int> & w2s, float label) {
  float lambda_ = 0.001;
  float pred = 0.0;
  vector<float> & f1 = model_[w1];
  vector<float> sf2(f1.size(), 0.0);
  
  for(int i = 0; i < w2s.size(); i++) {
    int w2 = w2s[i];
    vector<float> & f2 = model_[w2];
    for(int j = 0; j < f2.size(); j++) {
      sf2[j] += f2[j];
    }
    pred += dot(f1, f2);
  }
  pred = 1.0 / (1.0 + exp(-1.0 * pred));
  float err = (label - pred);
  
  vector<float> d1(f1.size(), 0.0f);
  for (int i = 0; i < f1.size(); i++) {
    float v1 = f1[i];
    float v2 = sf2[i];
    d1[i] += learning_rate_ * (err * v2 - lambda_);
  }

  for(int i = 0; i < w2s.size(); i++) {
    int w2 = w2s[i];
    vector<float> & f2 = model_[w2];
    vector<float> d2(f2.size(), 0.0f);
  
    for(int i = 0; i < f1.size(); i++) {
      float v1 = f1[i];
      float v2 = f2[i];
      
      d2[i] += learning_rate_ * (err * v1 - lambda_);
    }

    Lock(w2);
    for(int i = 0; i < f2.size(); i++) {
      if (f2[i] * (f2[i] + d2[i]) < 0.0)
        f2[i] = 0.0;
      else:
        f2[i] += d2[i];
    }
    UnLock(w2);
  }
  Lock(w1);
  for(int i = 0; i < f1.size(); i++) {
    if (f1[i] * (f1[i] + d1[i]) < 0.0)
      f1[i] = 0.0;
    else:
      f1[i] += d1[i];
  }
  UnLock(w1);
  
  return err;
}

}

