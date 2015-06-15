#include "functions.h"
#include "Python.h"
#include <cmath>
#include <iostream>
#include <thread>

using namespace std;

const double PI = std::atan(1.0)*4;

double easy_test(double x[])
{
  return (5.5 - x[0]) * (7.5 - x[1]);
}

double easy_test2(double x[])
{
  return sphere(x, 1);
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
double sphere_5(double x[])
{
  return sphere(x, 3);
}

double sphere_10(double x[])
{
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  return sphere(x, 6);
}

double sphere_15(double x[])
{
  return sphere(x, 9);
}

double sphere_20(double x[])
{
  return sphere(x, 12);
}

double sphere_25(double x[])
{
  return sphere(x, 15);
}

double sphere_30(double x[])
{
  return sphere(x, 30);
}

double sphere_4(double x[])
{
  return sphere(x, 4);
}

double sphere_3(double x[])
{
  //std::this_thread::sleep_for(std::chrono::milliseconds(600));
  return sphere(x, 3);
  //double result = sphere(x, 3);
  ////std::this_thread::sleep_for(std::chrono::milliseconds(500 + (int) result));
  //return result;
}

double sphere(double x[], int n)
{
  double sum = 0;
  for(int i = 0; i < n; i++) {
    sum = sum + pow(x[i],2);
  }
  return sum;
}

double prob09(double x[])
{
  return 100*pow(x[1] - pow(x[0], 2), 2) + pow(1 - x[0], 2);
}

double tang(double x[])
{
  double sum = 0;
  for(int i = 0; i < 2; i++) {
    sum += pow(x[i], 4) - 16 * pow(x[i], 2) + 5 * x[i];
  }
  return sum / 2;
}
