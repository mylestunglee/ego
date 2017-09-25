#pragma once
#include <string>
#include "functions.hpp"
#include "ego.hpp"

using namespace std;

enum class Mode {
	undefined,
	optimise,
	transfer,
	compare
};

int main(int argc, char* argv[]);

void print_help(ostream& cstr);

void simulate_results(EGO& ego, results_t results);
