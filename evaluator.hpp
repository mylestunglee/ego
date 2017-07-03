#pragma once

#include <string>
#include <vector>
#include <map>
#include <mutex>

using namespace std;

class Evaluator {
	public:
	    Evaluator(string script);
	    vector<double> evaluate(vector<double> x);
		void save(string filename);
		bool has_evaluated(vector<double> x);
	private:
	    string script;
	    map<vector<double>, vector<double>> cache;
	    vector<string> execute(string command);
		mutex cache_lock;
};
