#pragma once
#include <vector>

using namespace std;

class Surrogate {
	public:
		virtual ~Surrogate();
		virtual void add(vector<double> x, double y) = 0;
		virtual double mean(vector<double> x) = 0;
		virtual double sd(vector<double> x) = 0;
		virtual void optimise() = 0;
		virtual double cross_validate() = 0;
};
