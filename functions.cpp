#include "functions.h"
#include <vector>
#include <cmath>
#include <iostream>

void test_evaluator(vector<double> x, double &fitness, int &label, int &cost) {
  cost = 10;
  label = 0;
  if (x[0] > 4) label = 2;
  if (x[0] > 4) label = 1;
  fitness = (x[0] - 1) * (x[0] - 1) * (x[1] + 3) * (x[1] + 3) + 1;
}

