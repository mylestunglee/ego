#include "evaluator.hpp"
#include "csv.hpp"
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <iostream>

using namespace std;

Evaluator::Evaluator(string script) {
	this->script = script;
}

// Consults proxy then packages x for execution
vector<double> Evaluator::evaluate(vector<double> x) {
	cache_lock.lock();
	if (cache.find(x) != cache.end()) {
		vector<double> result = cache[x];
		cache_lock.unlock();
		return result;
	}
	cache_lock.unlock();

	string command = script;

	for (double arg : x) {
		command += " " + to_string(arg);
	}

	vector<double> result;

	for (string line : execute(command)) {
		try {
			result.push_back(stof(line));
		} catch (const invalid_argument& ia) {
			cout << line;
		}
	}

	cache_lock.lock();
	cache[x] = result;
	cache_lock.unlock();

    return result;
}

// Executes command returning a vector of lines from stdout
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

// Saves the cache as a CSV file
void Evaluator::save(string filename) {
	vector<vector<string>> data;
	for (pair<vector<double>, vector<double>> pair : cache) {
		vector<string> line;
		for (double x : pair.first) {
			line.push_back(to_string(x));
		}
		for (double y : pair.second) {
			line.push_back(to_string(y));
		}
		data.push_back(line);
	}
	write(filename, data);
}

// Simulates an execution without evalauting the script
void Evaluator::simulate(vector<double> x, vector<double> y) {
	cache_lock.lock();
	cache[x] = y;
	cache_lock.unlock();
}
