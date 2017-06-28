#pragma once

#include <string>
#include <vector>
#include <map>

using namespace std;

class Evaluator {
  public:
    Evaluator(string script);
    vector<double> evaluate(vector<double> x);
	void save(string filename);
  private:
    string script;
    map<vector<double>, vector<double>> cache;
    vector<string> execute(string command);
};
