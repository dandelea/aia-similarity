import argparse

import warnings
warnings.filterwarnings("ignore")

from kmeans import task01
from text_similarity import task02
from usenet import task03_pre, task03_run

import time

def run(task, operation):
	if task==1:
		t = time.time()
		task01(3, True)
		print("-------- Process finished in " + str(time.time() - t) + " seconds. ------------")
	elif task==2:
		t = time.time()
		task02()
		print("-------- Process finished in " + str(time.time() - t) + " seconds. ------------")
	else:
		k = int(input('Input k: '))
		if operation=='pre':
			t = time.time()
			task03_pre(k)
			print("-------- Process finished in " + str(time.time() - t) + " seconds. ------------")
		else:
			query = input('Query input: ')
			t = time.time()
			task03_run(query, k, 10)
			print("-------- Process finished in " + str(time.time() - t) + " seconds. ------------")


if __name__=='__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-t", "--task", required = True, choices=['1','2','3','01','02','03'], 
		help = "Task number: 1, 2 or 3.")
	ap.add_argument("-o", "--operation", required = False, choices=['pre', 'run'],
		help = "Operation : pre/run")
	args = vars(ap.parse_args())
	task = int(args['task']) or 5
	if task==3 and args['operation'] is None:
		raise ValueError('Operation input is mandatory')
	run(task, args['operation'])