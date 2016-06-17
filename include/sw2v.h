#ifndef SW2V_H_
#define SW2V_H_

#include <vector>
#include <map>
#include <string>
using namespace std;

#include <omp.h>

#define MAX(x, y) (x) > (y) ? (x) : (y)
#define MIN(x, y) (x) < (y) ? (x) : (y)
#define NLOCK    512

namespace sw2v {
  
class SparseWord2Vec {

public:
SparseWord2Vec(int nhidden, float learning_rate)
    : nhidden_(nhidden), learning_rate_(learning_rate) {
    locks_ = vector<omp_lock_t>(NLOCK);
    for(int i = 0; i < NLOCK; i++)
      omp_init_lock(&locks_[i]);
  }

  void Lock(int w) {
    omp_set_lock(&locks_[w % NLOCK]);
  }

  void UnLock(int w) {
    omp_unset_lock(&locks_[w % NLOCK]);
  }
  
  void LoadData(const char * fname);
  void InitModel();
  void Train();
  float OneStep(int w1, vector<int> & w2, float label);
  void PrintHiddens();
  void SaveModel();
private:
  vector<int> data_;
  map<string, int> word_index_;
  vector<string> words_;
  vector< vector<float> > model_;
  vector<omp_lock_t> locks_;
  int nhidden_;
  float learning_rate_;
};
  
} // namespace sw2v

#endif // SW2V_H_
