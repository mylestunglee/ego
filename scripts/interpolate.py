import numpy as np
from sklearn import gaussian_process
import csv
import sys

if len(sys.argv) < 3:
	print('Usage: interpolate.py results.csv config.txt [x_1, ...]')
	sys.exit(1)

# Read number of parameters
config_csv_file = open(sys.argv[2])
config_spam_reader = csv.reader(config_csv_file, delimiter = ',')
index = 0
for row in config_spam_reader:
	if len(row) == 0 or len(row[0]) == 0 or row[0][0] == '#':
		continue
	# 7th row contains lower boundaries in config file
	if index == 6:
		domain = len(row)
		break
	index += 1

config_csv_file.close()

if len(sys.argv) != 3 + domain:
	print('Invalid number of parameters to interpolate')
	sys.exit(1)

# Read old results and split CSV columns into domain and range columns
results_csvfile = open(sys.argv[1])
results_spamreader = csv.reader(results_csvfile, delimiter=',')
xs = []
ys = []

for row in results_spamreader:
	xs.append(list(map(float, row[:domain])))
	ys.append(list(map(float, row[domain:])))

results_csvfile.close()

# Interpolate results
gp = gaussian_process.GaussianProcessRegressor()
gp.fit(xs, ys)

x = list(map(float, sys.argv[3:]))

for zs in gp.predict([x]):
	for i in range(len(zs)):
		# Exit code printed as an integer
		if i == 1:
			print(int(round(zs[i])))
		else:
			print(zs[i])
