#include <cstdlib>
#include <ctime>
using namespace std;

#include <omp.h>
#include <sw2v.h>

int main(int argc, char ** argv) {
  srand(time(NULL));
  omp_set_num_threads(16);
  sw2v::SparseWord2Vec algo(100, 0.01);
  algo.LoadData("./data/text8");
  algo.InitModel();
  algo.Train();
}
