#include <assert.h>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <gsl_randist.h>
#include <gsl_cdf.h>
#include <gsl_statistics.h>
#include <gsl_vector.h>
#include <gsl_multifit.h>
#include <gsl_multimin.h>
#include "ihs.hpp"
#include "functions.hpp"

// Returns true iff x is bounded inside an n-dimensional hypercube defined by boundaries
bool is_bounded(vector<double> x, boundaries_t boundaries) {
	if (boundaries.empty()) {
		return false;
	}

	assert(x.size() == boundaries.size());

	for (size_t i = 0; i < x.size(); i++) {
		assert(boundaries[i].first <= boundaries[i].second);
		if (x[i] < boundaries[i].first || x[i] > boundaries[i].second) {
			return false;
		}
	}

	return true;
}

// Returns a sample from a uniform distribution of a hypercube defind by boundaries
vector<double> generate_uniform_sample(gsl_rng* rng, boundaries_t boundaries) {
	vector<double> sample;

	for (auto boundary : boundaries) {
		assert(boundary.first <= boundary.second);
		sample.push_back(gsl_ran_flat(rng, boundary.first, boundary.second));
	}

	return sample;
}

// Returns the Euclidean distance between two points
double euclidean_distance(vector<double> x, vector<double> y) {
	assert (x.size() == y.size());
	double distance = 0.0;
	for (size_t i = 0; i < x.size(); i++) {
		distance += pow(x[i] - y[i], 2);
	}
	return sqrt(distance);
}

// Pretty prints a vector
void print_vector(vector<double> x) {
	cout << "(" << setprecision(3);
	for (size_t i = 0; i < x.size(); i++) {
		cout << x[i];
		if (i < x.size() - 1) {
			cout << ",";
		}
	}
	cout << ")";
}

// Finds the probability of success given a normal distribution
double success_probability(double mean, double sd) {
	double upper = gsl_cdf_gaussian_P(1.5 - mean, sd);
	double middle = gsl_cdf_gaussian_P(0.5 - mean, sd);
	double lower = gsl_cdf_gaussian_P(-0.5 - mean, sd);
	double success = upper - middle;
	double normal = upper - lower;
	return success / normal;
}

// Returns the hypercuboid intersection of the two hypercuboid bxs and bys
boundaries_t get_intersection(boundaries_t bxs, boundaries_t bys) {
	assert(bxs.size() == bys.size());
	boundaries_t result;
	for (size_t i = 0; i < bxs.size(); i++) {
		double lower = max(bxs[i].first, bys[i].first);
		double upper = min(bxs[i].second, bys[i].second);
		if (lower > upper) {
			return {};
		} else {
			result.push_back(make_pair(lower, upper));
		}
	}
	return result;
}

// Generates a Latin hypercuboid bounded by boundaries with samples samples
vector<vector<double>> generate_latin_samples(gsl_rng* rng, size_t samples,
	boundaries_t boundaries) {
	size_t dimension = boundaries.size();
	int seed = gsl_rng_get(rng);
	int* latin = ihs(dimension, samples, 5, seed);
	assert(latin != NULL);

	vector<vector<double>> xs;

	// Scale latin hypercube to fit parameter space
	for (size_t i = 0; i < samples; i++) {
		vector<double> x;
		for (size_t j = 0; j < dimension; j++) {
			double lower = boundaries[j].first;
			double upper = boundaries[j].second;
			double x_j = lower + (latin[i * dimension + j] - 1.0) /
				(samples - 1.0) * (upper - lower);
			x.push_back(x_j);
		}
		xs.push_back(x);
	}

	delete latin;

	return xs;
}

// Returns the boundaries that bounds all xs in results
boundaries_t infer_boundaries(results_t results) {
	assert(!results.empty());

	vector<vector<double>> xs(results[0].first.size(), vector<double>());
	for (auto result : results) {
		auto x = result.first;
		for (size_t i = 0; i < x.size(); i++) {
			xs[i].push_back(x[i]);
		}
	}

	boundaries_t boundaries;
	for (auto x : xs) {
		auto boundary = minmax_element(x.begin(), x.end());
		boundaries.push_back(make_pair(x[boundary.first - x.begin()],
			x[boundary.second - x.begin()]));
	}

	return boundaries;
}

// Given a polynomial of coefficents c_0, c_1..., compute c_0 + c_1x + c_2x^2...
double apply_polynomial(double x, vector<double> coeffs) {
	double sum = 0.0;
	for (size_t i = 0; i < coeffs.size(); i++) {
		sum += coeffs[i] * pow(x, i);
	}
	return sum;
}

// Returns true iff all points in bxs are bounded by bys
bool is_subset(boundaries_t bxs, boundaries_t bys) {
	assert(bxs.size() == bys.size());
	for (size_t i = 0; i < bxs.size(); i++) {
		if (bxs[i].first < bys[i].first || bxs[i].second > bys[i].second) {
			return false;
		}
	}
	return true;
}

// Fits a set of 2D points to a N-dimensional polynomial fit
vector<double> fit_polynomial(vector<double> xs, vector<double> ys, int degree)
{
    assert(xs.size() == ys.size());

    gsl_multifit_linear_workspace *ws;
    gsl_matrix* cov;
    gsl_matrix* X;
    gsl_vector* y;
    gsl_vector* c;
    double chisq;
    int order = degree + 1;
    int n = xs.size();
    X = gsl_matrix_alloc(n, order);
    y = gsl_vector_alloc(n);
    c = gsl_vector_alloc(order);
    cov = gsl_matrix_alloc(order, order);

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < order; j++) {
            gsl_matrix_set(X, i, j, pow(xs[i], j));
        }
        gsl_vector_set(y, i, ys[i]);
    }

    ws = gsl_multifit_linear_alloc(n, order);
    gsl_multifit_linear(X, y, c, cov, &chisq, ws);

    vector<double> coeffs;

    for (int i = 0; i < order; i++) {
        coeffs.push_back(gsl_vector_get(c, i));
    }

    gsl_multifit_linear_free(ws);
    gsl_matrix_free(X);
    gsl_matrix_free(cov);
    gsl_vector_free(y);
    gsl_vector_free(c);

    return coeffs;
}

// Calculates Pearson and Spearman correlation coefficents
void calc_correlation(vector<double> xs, vector<double> ys,
    double &pearson, double& spearman) {
    assert(xs.size() == ys.size());

    size_t n = xs.size();
        const size_t stride = 1;
    gsl_vector_const_view gsl_x = gsl_vector_const_view_array(&xs[0], n);
    gsl_vector_const_view gsl_y = gsl_vector_const_view_array(&ys[0], n);
    pearson = gsl_stats_correlation(
        (double*) gsl_x.vector.data, stride,
        (double*) gsl_y.vector.data, stride, n);
    double work[2 * n];
    spearman = gsl_stats_spearman(
        (double*) gsl_x.vector.data, stride,
        (double*) gsl_y.vector.data, stride, n, work);
}

// Returns true iff a boundary is suitable for knowledge transfer
bool are_valid_boundaries(boundaries_t boundaries) {
	for (auto boundary : boundaries) {
		if (boundary.first >= boundary.second) {
			return false;
		}
	}
	return true;
}

// Computes hypervolume of a hypercuboid defined by boundaries
double calc_hypervolume(boundaries_t boundaries) {
	assert(are_valid_boundaries(boundaries));

	double hypervolume = 1.0;
	for (auto boundary : boundaries) {
		hypervolume *= boundary.second - boundary.first;
	}
	return hypervolume;
}

// Zips two vectors in a vector of pairs
vector<pair<double, double>> read_boundaries(vector<string> xs, vector<string> ys) {
	vector<pair<double, double>> result;
	for (size_t i = 0; i < min(xs.size(), ys.size()); i++) {
		result.push_back(make_pair(stof(xs[i]), stof(ys[i])));
	}
	return result;
}

// Rounds each value of x to the nearest integer
vector<double> round_vector(vector<double> x) {
	vector<double> result;
	for (auto v : x) {
		result.push_back(round(v));
	}
	return result;
}

// Given a hypercuboid of dimension n with density d, generates evenly
// distributed with d ^ n points
vector<vector<double>> generate_grid_samples(size_t density,
	boundaries_t boundaries) {
	assert(density >= 2);

	if (boundaries.empty()) {
		return {{}};
	}
	vector<vector<double>> result;
	auto boundary = boundaries.back();
	auto lower = boundary.first;
	auto upper = boundary.second;
	boundaries.pop_back();
	vector<vector<double>> subresults = generate_grid_samples(density,
		 boundaries);
	for (size_t sample = 0; sample < density; sample++) {
		for (auto subresult : subresults) {
			subresult.push_back(lower + (upper - lower) * (sample /
				(density - 1.0)));
			result.push_back(subresult);
		}
	}
	return result;
}

vector<double> join_vectors(vector<double> x, vector<double> y) {
	vector<double> result;
	result.reserve(x.size() + y.size());
	result.insert(result.end(), x.begin(), x.end());
	result.insert(result.end(), y.begin(), y.end());
	return result;
}

// Given two vectors x and y, compute x ++ y
boundaries_t join_boundaries(boundaries_t x, boundaries_t y) {
	boundaries_t result;
	result.reserve(x.size() + y.size());
	result.insert(result.end(), x.begin(), x.end());
	result.insert(result.end(), y.begin(), y.end());
	return result;
}

// Finds a local minimum of func(x, arg) with the initial point x_0
vector<double> minimise_local(double (*func)(const gsl_vector*, void*),
    void* arg, vector<double> x, double convergence_threshold,
    size_t max_trials, double& minimum) {

	const gsl_multimin_fminimizer_type *algorithm = gsl_multimin_fminimizer_nmsimplex2;
    gsl_multimin_function gsl_func;

    // Starting point
	size_t dimension = x.size();
    gsl_vector* gsl_x = gsl_vector_alloc (dimension);
    for (size_t i = 0; i < dimension; i++) {
        gsl_vector_set(gsl_x, i, x[i]);
    }

    // Set initial step sizes to 1
    gsl_vector* ss = gsl_vector_alloc (dimension);
    gsl_vector_set_all (ss, 1.0);

    // Initialize method and iterate
	gsl_func.n = dimension;
	gsl_func.f = func;
	gsl_func.params = arg;

    gsl_multimin_fminimizer* s = gsl_multimin_fminimizer_alloc (algorithm, dimension);
    gsl_multimin_fminimizer_set (s, &gsl_func, gsl_x, ss);

    for (size_t trial = 0; trial < max_trials; trial++) {
        int status = gsl_multimin_fminimizer_iterate(s);

        if (status) {
            break;
        }

        double size = gsl_multimin_fminimizer_size (s);
        status = gsl_multimin_test_size (size, convergence_threshold);

        if (status != GSL_CONTINUE) {
            break;
        }
    }

    vector<double> result;

    for (size_t i = 0; i < dimension; i++) {
        result.push_back(gsl_vector_get(s->x, i));
    }

    minimum = s->fval;

    gsl_vector_free(gsl_x);
    gsl_vector_free(ss);
    gsl_multimin_fminimizer_free (s);

    return result;
}

// Runs multiple trials to maximise improvement for global minimum
vector<double> minimise(double (*func)(const gsl_vector*, void*),
    vector<double> (*gen)(void*), void* arg, double convergence_threshold,
    size_t max_trials, double& minimum) {
    vector<double> x_best;
    for (size_t trial = 0; trial < max_trials; trial++) {
        double minimum_trial = 0.0;
        auto x = minimise_local(func, arg, gen(arg), convergence_threshold,
			max_trials, minimum_trial);
        if (minimum_trial < minimum) {
            minimum = minimum_trial;
            x_best = x;
        }
    }
    return x_best;
}

// Returns true iff the evaluation result passes all constraints
bool is_success(vector<double> y, size_t constraints, size_t costs) {
	const size_t FITNESS_LABEL_OFFSET = 2;
	const size_t LABEL_INDEX = 1;
	assert(y.size() == FITNESS_LABEL_OFFSET + constraints + costs);
	if (y[LABEL_INDEX] != 0.0) {
		return false;
	}
	for (size_t constraint = 0; constraint < constraints; constraint++) {
		if (y[FITNESS_LABEL_OFFSET + constraint] > 1.0) {
			return false;
		}
	}
	return true;
}

// Converts a GSL vector into a std::vector
vector<double> gsl_to_std_vector(const gsl_vector* v) {
	vector<double> x;
	for (size_t i = 0; i < v->size; i++) {
		x.push_back(gsl_vector_get(v, i));
	}
	return x;
}
