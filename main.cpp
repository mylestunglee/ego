#include "ego.h"
#include "surrogate.h"
#include "functions.h"
#include "optimise.h"
#include <vector>

using namespace std;

int main(int argc, char * argv[]) {
  int dimension = 2;

  vector<double> lower = {-5, -5};
  vector<double> upper = {5, 5};

  Surrogate *sg = new Surrogate(dimension, SEiso);

  EGO *ego = new EGO(dimension, sg, lower, upper, quadratic);

  ego->search_type = 1;
  ego->max_fitness = 0;
  ego->is_discrete = true;
  ego->n_sims = 50;
  ego->max_points = upper[0] - lower[0];
  ego->num_points = ego->max_points;
  ego->pso_gen = 100;
  ego->exhaustive = true;

  ego->latin_hypercube(5,5);
  ego->sg->train();

  opt *op = new opt(2, upper, lower, ego, false);
  vector<vector<double>> best = op->combined_optimise({2, 2}, 10, 10, 10);
  for (vector<double> group : best) {
    for (double elem : group) {
      cout << elem << '\t';
    }
    cout << endl;
  }
}
