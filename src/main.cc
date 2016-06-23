#include <cstdlib>
#include <ctime>
using namespace std;

#include <omp.h>
#include <sw2v.h>

int main(int argc, char ** argv) {
  srand(time(NULL));
  omp_set_num_threads(16);
  sw2v::SparseWord2Vec algo(200, 5, 16, 0.005);
  algo.LoadVocab("./data/text8.vocab");
  algo.InitModel();
  for(int i = 0; i < 100; i++) {
    DataIter iter("./data/text8.ints");
    algo.Train(iter);
    cout << "epoc " << i << endl;
  }
}
