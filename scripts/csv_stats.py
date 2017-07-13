import numpy as np
from tabulate import tabulate
from sklearn import gaussian_process
import csv
import sys

if len(sys.argv) != 2:
	print('Usage: csv_stats.py data.csv')
	sys.exit(1)

csvfile = open(sys.argv[1])
spamreader = csv.reader(csvfile, delimiter=',')

rows = []

for row in spamreader:
	rows.append(row)

columns = zip(*rows)
table = []

for column in columns:
	data = list(map(float, column))
	table.append([min(data), max(data), len(set(data))])

print(tabulate(table, ['min', 'max', 'unique elements'], showindex='always'))

