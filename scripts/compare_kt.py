import math
import numpy as np
import csv
import sys

if len(sys.argv) != 3:
	print('Usage: plot_compare.py results_no_kt results_kt')
	sys.exit(1)

def read_csv(filename):
	csvfile = open(filename, 'r')
	spamreader = csv.reader(csvfile, delimiter=',')

	rows = []
	for row in spamreader:
		rows.append(list(map(float, row)))
	return rows

# Read data from optimiser
no_kt = read_csv(sys.argv[1])
kt = read_csv(sys.argv[2])

# Find minimum and maximum
flatten = lambda l: [item for sublist in l for item in sublist]
is_defined = lambda x: x < 10 ** 308
remove_undefineds = lambda l: filter(is_defined, l)
ys = flatten(no_kt) + flatten(kt)
y_min = min(ys)
y_max = max(remove_undefineds(ys))

# Calculates efficency of optimisation
def calc_score(vs, m):
	sum = 0.0
	curr = y_max
	for i in range(m):
		if i < len(vs) and is_defined(vs[i]):
			curr = vs[i]
		sum += curr - y_min
	return sum / (m * (y_max - y_min))

m = math.floor(np.mean(list(map(len, no_kt + kt))))
calc_score_fixed = lambda vs: calc_score(vs, m)

scores_no_kt = list(map(calc_score_fixed, no_kt))
scores_kt = list(map(calc_score_fixed, kt))

scores_no_kt_mean = np.mean(scores_no_kt)
scores_no_kt_sd   = np.std (scores_no_kt)
scores_kt_mean    = np.mean(scores_kt   )
scores_kt_sd      = np.std (scores_kt   )
improvement_mean  = scores_no_kt_mean / scores_kt_mean

ratios = [x / y for x in scores_no_kt for y in scores_kt if y > 0.0]

ratio_mean = np.mean(ratios)
ratio_sd   = np.std (ratios)

improvement_sd = (improvement_mean / ratio_mean) * ratio_sd

print('No KT optimisation score mean: %0.3f' % scores_no_kt_mean)
print('No KT optimisation score sd:   %0.3f' % scores_no_kt_sd  )
print('KT optimisation score mean:    %0.3f' % scores_kt_mean   )
print('KT optimisation score sd:      %0.3f' % scores_kt_sd     )
print('Improvement ratio mean:        %0.3f' % improvement_mean )
print('Improvement ratio sd:         ~%0.3f' % improvement_sd   )
