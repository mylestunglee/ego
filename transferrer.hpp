#include <string>
#include <vector>
#include "evaluator.hpp"

using namespace std;

class Transferrer {
	public:
		Transferrer(
			string filename_results_old,
			string filename_script_new,
			double sig_level);
		~Transferrer();
		void transfer();
	private:
		Evaluator* evaluator;
		vector<pair<vector<double>, vector<double>>> results_old;
		vector<pair<double, double>> boundaries;
		double sig_level;

		void read_results(string filename);
		void calc_correlation(
			vector<double> x,
			vector<double> y,
			double &pearson,
			double &spearman);
		vector<pair<vector<double>, vector<double>>> sample_results_old();
		vector<double> fit_polynomial(
			vector<double> xs,
			vector<double> ys,
			int degree);
		static bool fitness_more_than(
			pair<vector<double>, vector<double>> xs,
			pair<vector<double>, vector<double>> ys);
		double calc_label_correlation(
			vector<pair<vector<double>, int>> sample_passes);

};
