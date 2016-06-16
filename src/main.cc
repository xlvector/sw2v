#include <cstdlib>
using namespace std;

#include <sw2v.h>

int main(int argc, char ** argv) {
  srand(time(NULL));
  sw2v::SparseWord2Vec algo(100);
  algo.LoadData("./data/text8.bin");
  algo.InitModel();
  algo.Train();
}
