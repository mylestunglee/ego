#include "ego.h"
#include "surrogate.h"
#include <vector>

using namespace std;

static double fit(double x[])
{
  return (7 - x[0]) * (7 - x[0]);
}

int main() {
  vector<double> lower = {0.0};
  vector<double> upper = {10.0};
  EGO ego(1, lower, upper, &fit);
  for(double i = 0; i < 3; ++i) {
    vector<double> x = {i * 5};
    double y = fit(&x[0]);
    ego.add_training(x, y);
  }
  ego.run();
  vector<double> r;
  for(int i = 1; i < 7; i++) {
    r = ego.max_ei_par(i);
    for(int j = 0; j < i; j++) {
      cout << r[j] << " ";
    } 
    cout << endl;
  }
  cout << ego.best_result()[0] << endl;
}

