#include "evaluator.hpp"
#include <memory>
#include <stdexcept>
#include <string>
#include <array>

using namespace std;

Evaluator::Evaluator(string script) {
	this->script = script;
}

/* Consults proxy then packages x for execution */
vector<double> Evaluator::evaluate(vector<double> x) {
	if (cache.find(x) != cache.end()) {
		return cache[x];
	}

	string command = script;

	for (double arg : x) {
		command += " " + to_string(arg);
	}

	vector<double> result;

	for (string line : execute(command)) {
		result.push_back(atof(line.c_str()));
	}

	cache[x] = result;

    return result;
}

/* Executes command returning a vector of lines from stdout */
vector<string> Evaluator::execute(string command) {
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

/* Saves the cache as a CSV file */
void save(string filename) {

}
