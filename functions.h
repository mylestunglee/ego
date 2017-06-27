#pragma once
#include <vector>

using namespace std;

typedef void (* evaluator_t)(vector<double> x, double &fitness, int &label, int &cost);

void test_evaluator(vector<double> x, double &fitness, int &label, int &cost);

double easy_test(double x[]);
double easy_test2(double x[]);
double ackley(double z[]);
double goldstein(double z[]);
double sphere(double x[], int n);
double sphere_4(double x[]);
double sphere_3(double x[]);
double sphere_2(double x[]);
double sphere_5(double x[]);
double sphere_10(double x[]);
double sphere_15(double x[]);
double sphere_20(double x[]);
double sphere_25(double x[]);
double sphere_30(double x[]);
double prob09(double x[]);
double tang(double x[]);

enum example {quad=1, pq, rtm};
