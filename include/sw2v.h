#ifndef SW2V_H_
#define SW2V_H_

#include <vector>
#include <map>
#include <string>
#include <cmath>
using namespace std;

#include <omp.h>
#include <data_iter.h>

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define NLOCK    8192

#define SIGMOID_MAX 6.0
#define SIGMOID_MIN -6.0

namespace sw2v {

struct Sample {
  Sample(float label, int target) {
    label_ = label;
    target_ = target;
  }
  float label_;
  int target_;
  vector<int> context_;
};
  
class SparseWord2Vec {
 public:
  SparseWord2Vec(int nhidden, int win_size, int batch_size, float learning_rate) {
    nhidden_ = nhidden;
    win_size_ = win_size;
    learning_rate_ = learning_rate;
    batch_size_ = batch_size;
    
    locks_ = vector<omp_lock_t>(NLOCK);
    for(int i = 0; i < NLOCK; i++)
      omp_init_lock(&locks_[i]);
    omp_init_lock(&lock_);

    sigmoid_ = vector<float>(10000, 0);
    float h = (SIGMOID_MAX - SIGMOID_MIN) / 10000.0;
    for(int i = 0; i < 10000; i++) {
      float x = i * h + SIGMOID_MIN;
      sigmoid_[i] = 1.0 / (1.0 + exp(-1.0 * x));
    }
  }

  float Sigmoid(const float & x) {
    if(x > SIGMOID_MAX) return 1.0;
    if(x < SIGMOID_MIN) return 0.0;
    return sigmoid_[(int)(10000.0 * (x - SIGMOID_MIN) / (SIGMOID_MAX - SIGMOID_MIN))];
  }

  void LockAll() {
    omp_set_lock(&lock_);
  }

  void UnLockAll() {
    omp_unset_lock(&lock_);
  }

  void Lock(int w) {
    omp_set_lock(&locks_[w % NLOCK]);
  }

  void UnLock(int w) {
    omp_unset_lock(&locks_[w % NLOCK]);
  }
  
  void LoadVocab(const char * fname);
  void InitModel();
  void Train(DataIter & iter);
  vector< pair<int, float> > MiniBatch(const vector<Sample> & batch);
  float OneStep(int w1, const vector<int> & w2, float label,
                const map<int, float*> & batch_model,
                map<int, vector<float> > & grads);
  void SaveModel();
private:
  vector<float> sigmoid_;
  vector<int> negative_;
  vector<string> words_;
  vector<int> freq_;
  vector<float> model_;
  vector<omp_lock_t> locks_;
  omp_lock_t lock_;
  int nhidden_;
  int batch_size_;
  int win_size_;
  float learning_rate_;
};
  
} // namespace sw2v

#endif // SW2V_H_
