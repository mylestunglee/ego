#include "ego.h"
#include "surrogate.h"
#include <vector>

using namespace std;
const double PI = std::atan(1.0)*4;

static double easy_test(double x[])
{
  return (5.5 - x[0]) * (7.5 - x[1]);
}

static double ackley(double z[])
{
  double x = z[0];
  double y = z[1];
  double part1 = -20 * exp(-0.2 * sqrt(0.5 * (x * x + y * y)));
  double part2 = -exp(0.5 * (cos(2 * PI * x) + cos(2 * PI * y))) + exp(1) + 20;
  return part1 + part2;
}

int main(int argc, char * argv[]) {

  double (*fitness)(double x[]) = &easy_test;
  int dimension = 2;
  vector<double> lower = {0.0, 0.0};
  vector<double> upper = {10.0, 10.0};
  int which = 1;

  if(argc >= 2) {
    which = atoi(argv[1]);
    switch(which) {
      case 1:
        fitness = &easy_test;
	dimension = 2;
        break;
      case 2:
        fitness = &ackley;
	dimension = 2;
        lower = {-5.0, -5.0};
        upper = {5.0, 5.0};
        break;
      case 3:
        break;
    }
  }
    
  Surrogate sg(dimension, SEiso);
  sg.set_params(0.3, 2);
  EGO ego(dimension, &sg, lower, upper, fitness);

  if(which == 1) {
    for(double i = 1; i < 3; ++i) {
      for(double z = 1; z < 3; ++z) {
        vector<double> x = {i * 5, z*5};
        double y = fitness(&x[0]);
        ego.add_training(x, y);
      }
    }
    ego.max_fitness = -33.75;
    ego.max_iterations = 50;
    ego.is_discrete = true;
  } else if(which == 2) {
    for(double i = -5.0; i <= 5; i += 2) {
      for(double z = -5.0; z <= 5; z += 2) {
        vector<double> x = {i, z};
        double y = fitness(&x[0]);
        ego.add_training(x, y);
      }
    }
    ego.max_fitness = 0;
    ego.max_iterations = 50;
    ego.is_discrete = true;
  }

  ego.run();

  vector<double> r;
  for(int i = 1; i < 7; i++) {
    r = ego.max_ei_par(i);
    for(int j = 0; j < i * ego.dimension; j++) {
      cout << r[j] << " ";
    } 
    cout << endl;
  }
  cout << "BEST: " << endl;
  for(int i = 0; i < ego.dimension; i++) {
    cout << ego.best_result()[i] << " ";
  }
  cout << ": best result " << ego.best_fitness << endl;
}

