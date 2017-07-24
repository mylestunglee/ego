import matplotlib.pyplot as plt
import numpy as np
import csv
import sys

if len(sys.argv) != 4:
	print('Usage: plot_experiment.py results_no_kt results_kt output')
	sys.exit(1)

def aggregate(filename):
	# Read CSV file
	csvfile = open(filename, 'r')
	spamreader = csv.reader(csvfile, delimiter=',')

	rows = []
	max_columns = 0
	for row in spamreader:
		rows.append(row)
		max_columns = max(max_columns, len(row))

	# Aggregate data into means and sds
	indices = []
	lowers = []
	means = []
	uppers = []
	# max_columns = min(max_columns, 25)

	for i in range(max_columns):
		values = []
		for row in rows:
			if i >= len(row) or float(row[i]) > 10.0 ** 308.0:
				continue
			values.append(float(row[i]))
		if values != []:
			indices.append(i + 1)
			mean = np.mean(values)
			sd = np.std(values)
			lower = mean - sd
			upper = mean + sd

			lowers.append(lower)
			means.append(mean)
			uppers.append(upper)
	return indices, lowers, means, uppers

ni, nl, nm, nu = aggregate(sys.argv[1])
ti, tl, tm, tu = aggregate(sys.argv[2])

# Plot graph
plt.fill_between(ni, nl, nu, facecolor = 'r', alpha = 0.25, edgecolor = 'none')
plt.fill_between(ti, tl, tu, facecolor = 'b', alpha = 0.25, edgecolor = 'none')
plt.plot(ni, nm, label = 'Without KT', color = 'r')
plt.plot(ti, tm, label = 'With KT', color = 'b')
plt.legend()
plt.xlabel('Evaluations')
plt.ylabel('Fitness')
plt.savefig(sys.argv[3])

