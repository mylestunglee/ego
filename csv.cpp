#include "csv.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

using namespace std;

// Writes a CSV file located as filename using a two-dimensional vector of strings
void write(string filename, vector<vector<string>> data) {
	ofstream file(filename);
	for (vector<string> line : data) {
		for (unsigned i = 0; i < line.size(); i++) {
			file << line[i];
			if (i < line.size() - 1) {
				file << ',';
			}
		}
		file << '\n';
	}
}

// Reads a CSV file located as filename as a two-dimensional vector of strings
vector<vector<string>> read(string filename) {
	ifstream file(filename);
	string line;
	vector<vector<string>> data;
	while (getline(file, line)) {
		// Skip comment lines
		if (line[0] == '#') {
			continue;
		}

		stringstream ss(line);
		string cell;
		vector<string> tokens;
		while (getline(ss, cell, ',')) {
			tokens.push_back(cell);
		}
		data.push_back(tokens);
	}

	return data;
}
