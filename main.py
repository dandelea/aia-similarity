import argparse

from kmeans import task01
from test_similarity import task02
from usenet import task03

def run(task):
	if task==1:
		task01()
	elif task==2:
		task02()
	else:
		task03()

if __name__=='__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-t", "--task", required = True, choices=['1','2','3','01','02','03'], 
		help = "Task number: 1, 2 or 3.")
	args = vars(ap.parse_args())
	task = int(args['task']) or 5
	run(task)