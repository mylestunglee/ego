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
	means = []
	sds = []
	for i in range(max_columns):
		values = []
		for row in rows:
			if i >= len(row) or float(row[i]) > 10.0 ** 308.0:
				continue
			values.append(float(row[i]))
		means.append(np.mean(values))
		sds.append(np.std(values))
	return means, sds

no_kt_means, no_kt_sds = aggregate(sys.argv[1])
kt_means, kt_sds = aggregate(sys.argv[2])

# Plot graph
plt.plot(no_kt_means, label = 'Mean without KT')
plt.plot(no_kt_sds, label = 'Standard deviation without KT')
plt.plot(kt_means, label = 'Mean with KT')
plt.plot(kt_sds, label = 'Standard deviation with KT')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.savefig(sys.argv[3])

