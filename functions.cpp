#include "functions.h"
#include <cmath>
#include <iostream>
#include <thread>

using namespace std;

const double PI = std::atan(1.0)*4;

double easy_test(double x[])
{
  return (5.5 - x[0]) * (7.5 - x[1]);
}

double ackley(double z[])
{
  double x = z[0];
  double y = z[1];
  double part1 = -20 * exp(-0.2 * sqrt(0.5 * (x * x + y * y)));
  double part2 = -exp(0.5 * (cos(2 * PI * x) + cos(2 * PI * y))) + exp(1) + 20;
  return part1 + part2;
}

double goldstein(double z[])
{
  double x = z[0];
  double y = z[1];
  double part1 = 1 + pow(x + y + 1, 2) * (19 - 14*x + 3*x*x - 14*y + 6*x*y + 3*y*y);
  double part2 = 30 + pow(2*x - 3*y, 2) * (18 - 32*x + 12*x*x + 48*y - 36*x*y + 27*y*y);
  return part1 * part2;
}

double sphere_4(double x[])
{
  std::this_thread::sleep_for(std::chrono::milliseconds(600));
  return sphere(x, 4);
}

double sphere_3(double x[])
{
  std::this_thread::sleep_for(std::chrono::milliseconds(600));
  return sphere(x, 3);
}

double sphere(double x[], int n)
{
  double sum = 0;
  for(int i = 0; i < n; i++) {
    sum = sum + pow(x[i],2);
  }
  return sum;
}
