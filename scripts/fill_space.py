import numpy as np
from tabulate import tabulate
from sklearn import gaussian_process
import csv
import sys

if len(sys.argv) < 2 or len(sys.argv) % 2 != 0:
	print('Usage: python3 csv_stats.py results.csv [l_1, u_1, ...]')
	sys.exit(1)

csvfile = open(sys.argv[1], 'r')
spamreader = csv.reader(csvfile, delimiter=',')

# builds boundaries
boundaries = []
unzipped = list(map(int, sys.argv[2:]))
for i in range(len(unzipped) // 2):
	boundaries.append((unzipped[i * 2], unzipped[i * 2 + 1]))

def generate_grid_samples(boundaries, i):
	if i >= len(boundaries):
		return [[]]

	boundary = boundaries[i]
	result = []
	for j in range(boundary[0], boundary[1] + 1):
		samples = generate_grid_samples(boundaries, i + 1)
		for sample in samples:
			sample.append(j)
			result.append(sample)
	return result

# generates all possible positions in the boundary space
grid = generate_grid_samples(boundaries[::-1], 0)

columns = 0
fill = []
for row in spamreader:
	fill.append(list(map(float,row[: len(boundaries)])))
	columns = len(row)

def diff(first, second):
	result = []
	for item in first:
		if item not in second:
			result.append(item)
	return result

# files points not defined
space = diff(grid, fill)
csvfile.close()
csvfile = open(sys.argv[1], 'a')
spamwriter = csv.writer(csvfile, delimiter=',')

# writes each point
for cell in space:
	row = cell + [0, 1] + [0] * (columns - len(boundaries) - 2)
	spamwriter.writerow(list(map(str, row)))

csvfile.close()
