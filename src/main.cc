#include <cstdlib>
#include <string>
#include <cstdio>
#include <ctime>
using namespace std;

#include <omp.h>
#include <sw2v.h>

#ifndef LOCAL
#include "ps/ps.h"

void StartServer() {
  if (!ps::IsServer()) return;
  auto server = new ps::KVServer<float>(0);
  server->set_request_handle(ps::KVServerDefaultHandle<float>());
  ps::RegisterExitCallback([server](){ delete server; });
}
#endif

int main(int argc, char ** argv) {
#ifndef LOCAL
  StartServer();
  ps::Start();

  if (ps::IsWorker()) {
#endif  
    srand(time(NULL));
    sw2v::SparseWord2Vec algo(200, 5, 8, 0.005);
    algo.LoadVocab("./data/text8.vocab");

#if LOCAL
    int rank = 0;
#else
    int rank = ps::MyRank();
#endif
    for(int i = 0; i < 100; i++) {
      string fname = "./data/text8.ints." + to_string(rank);
      cout << fname << endl;
      DataIter iter(fname.c_str());
      algo.Train(iter);
      cout << "epoc " << i << endl;
    }
#ifndef LOCAL
  }
  
  ps::Finalize();
#endif
}
