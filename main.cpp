#include "ego.h"
#include "surrogate.h"
#include "functions.h"
#include "optimise.h"
#include <vector>

using namespace std;

int main(int argc, char * argv[]) {
  int dimension = 2;

  vector<double> lower = {-100, -100};
  vector<double> upper = {100, 100};

  Surrogate *sg = new Surrogate(dimension, SEiso, true, true);

  EGO *ego = new EGO(dimension, sg, lower, upper, "", 1);
  ego->evaluator = test_evaluator;

  /*ego->search_type = 1;
  ego->max_fitness = 0;
  ego->is_discrete = true;
  ego->n_sims = 50;
  ego->max_points = upper[0] - lower[0];
  ego->num_points = ego->max_points;
  ego->pso_gen = 100;
  ego->exhaustive = true;*/

  ego->sample_plan(10 * dimension, 5);
  ego->sg->train();
  cout << "Best fitness: ";
  for (double x : ego->best_result()) {
    cout << x << ", ";
  }
  cout << endl;

  /*opt *op = new opt(2, upper, lower, ego, false);
  vector<vector<double>> best = op->combined_optimise({2, 2}, 10, 10, 10);
  for (vector<double> group : best) {
    for (double elem : group) {
      cout << elem << '\t';
    }
    cout << endl;
  }*/

  /*for(int i = lower[0]; i <= upper[0]; i++) {
    for(int j = lower[1]; j <= upper[1]; j++) {
      ego->python_eval({(double) i, (double) j});

    }
  }*/

  delete ego;
}
