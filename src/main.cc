#include <cstdlib>
#include <ctime>
using namespace std;

#include <omp.h>
#include <sw2v.h>
#include "ps/ps.h"

void StartServer() {
  if (!ps::IsServer()) return;
  auto server = new ps::KVServer<float>(0);
  server->set_request_handle(ps::KVServerDefaultHandle<float>());
  ps::RegisterExitCallback([server](){ delete server; });
}

int main(int argc, char ** argv) {
  StartServer();
  ps::Start();

  if (ps::IsWorker()) {
    srand(time(NULL));
    sw2v::SparseWord2Vec algo(200, 5, 16, 0.005);
    algo.LoadVocab("./data/text8.vocab");
    algo.InitModel();
    for(int i = 0; i < 100; i++) {
      DataIter iter("./data/text8.ints");
      algo.Train(iter);
      cout << "epoc " << i << endl;
    }
  }

  ps::Finalize();
}
