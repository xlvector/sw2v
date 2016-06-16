#ifndef SW2V_H_
#define SW2V_H_

#include <vector>
#include <map>
using namespace std;

#define MAX(x, y) (x) > (y) ? (x) : (y)
#define MIN(x, y) (x) < (y) ? (x) : (y)

namespace sw2v {

  
class SparseWord2Vec {
 private:
  vector<int> data_;
  vector< vector<float> > model_;
  int nhidden_;

public:
SparseWord2Vec(int nhidden)
      : nhidden_(nhidden) {}
  
  void LoadData(const char * fname);
  void InitModel();
  void Train();
  float OneStep(int w1, int w2, float label);
};
  
} // namespace sw2v

#endif // SW2V_H_
