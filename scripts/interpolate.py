import numpy as np
from sklearn import gaussian_process
import csv
import sys

if len(sys.argv) < 3:
	print('Usage: interpolate.py data.csv domain [x_1, ...]')
	sys.exit(1)

csvfile = open(sys.argv[1])
spamreader = csv.reader(csvfile, delimiter=',')
xs = []
ys = []
domain = int(sys.argv[2])

for row in spamreader:
	xs.append(list(map(float, row[:domain])))
	ys.append(list(map(float, row[domain:])))

gp = gaussian_process.GaussianProcessRegressor()
gp.fit(xs, ys)

x = list(map(float, sys.argv[3:]))

for zs in gp.predict([x]):
	for i in range(len(zs)):
		if i == 1:
			print(int(round(zs[i])))
		else:
			print(zs[i])
