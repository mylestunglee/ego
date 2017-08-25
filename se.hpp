#pragma once

#include <string>
#include "evaluator.hpp"
#include "functions.hpp"

using namespace std;

class ScriptEvaluator : public Evaluator {
	public:
		ScriptEvaluator(string script);
		~ScriptEvaluator();
	private:
	    string script;
		point_t compute(point_t x);
	    vector<string> execute(string command);
};
