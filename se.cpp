#include "se.hpp"
#include "functions.hpp"
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <iostream>

using namespace std;

ScriptEvaluator::ScriptEvaluator(string script) {this->script = script;}

ScriptEvaluator::~ScriptEvaluator() {}

// Consults proxy then packages x for execution
point_t ScriptEvaluator::compute(point_t x) {
	string command = script;

	for (double arg : x) {
		command += " " + to_string(arg);
	}

	point_t y;

	for (string line : execute(command)) {
		try {
			y.push_back(stof(line));
		} catch (const invalid_argument& ia) {
			cout << line;
		}
	}

	return y;
}

// Executes command returning a vector of lines from stdout
vector<string> ScriptEvaluator::execute(string command) {
	array<char, 128> buffer;
	vector<string> result;
	std::shared_ptr<FILE> pipe(popen(command.c_str(), "r"), pclose);
	if (!pipe) throw std::runtime_error("popen() failed!");
	while (!feof(pipe.get())) {
		if (fgets(buffer.data(), 128, pipe.get()) != NULL)
			result.push_back(buffer.data());
	}
	return result;
}
