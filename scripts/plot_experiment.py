import matplotlib.pyplot as plt
import numpy as np
import csv
import sys

if len(sys.argv) < 4 or len(sys.argv) > 5:
	print('Usage: plot_experiment.py results_no_kt results_kt output [max_evalutions]')
	sys.exit(1)

def aggregate(filename):
	# Read CSV file
	csvfile = open(filename, 'r')
	spamreader = csv.reader(csvfile, delimiter=',')

	rows = []
	max_columns = 0
	for row in spamreader:
		cleaned = list(filter(None, row))
		rows.append(cleaned)
		max_columns = max(max_columns, len(cleaned))

	# Aggregate data into means and sds
	indices = []
	lowers = []
	means = []
	uppers = []
	if len(sys.argv) == 5:
		max_columns = min(max_columns, int(sys.argv[4]))

	for i in range(max_columns):
		values = []
		for row in rows:
			value = 10.0 ** 308.0
			if i >= len(row):
				value = float(row[-1])
			else:
				value = float(row[i])

			if value >= 10.0 ** 308.0:
				continue

			values.append(value)
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

