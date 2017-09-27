# Aggregates a CSV file to average a value over duplicated multi-dimensional
# function

import csv
import sys

if len(sys.argv) < 3:
	print('Usage: python3 parser.py results.csv dimension')
	exit(1)

csvfile = open(sys.argv[1], 'r')
spamreader = csv.reader(csvfile, delimiter = ',')
dimension = int(sys.argv[2])

data = {}

def is_number(s):
	try:
		float(s)
		return True
	except ValueError:
		return False

def is_all_numbers(ss):
	for s in ss:
		if not is_number(s):
			return False
	return True

for row in spamreader:
	if not is_all_numbers(row):
		continue

	xs = tuple(map(int, row[:dimension]))
	y = float(row[dimension])
	if xs in data:
		(n, sum) = data[xs]
		data[xs] = (n + 1, sum + y)
	else:
		data[xs] = (1, y)

nubbed = []
for key in data:
	(n, sum) = data[key]
	nubbed.append(list(key) + [sum / n])

sort = sorted(nubbed)

for row in sort:
	print(','.join(list(map(str, row))))
