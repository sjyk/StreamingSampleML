#!/usr/bin/env python

import numpy as np
import random
from cvxopt import solvers, matrix, mul
from sampleml import *

"""
This loads the smallest load the smallest dataset that we 
have. The housing dataset. This dataset is completely numerical
and is meant for regression tasks.
"""
def loadHousingDataSet():
	lineCount = 0
	data_matrix = np.zeros((0,14))

	with open('data/housing.csv') as file:
		line = file.readline()
		while line != "":
			example_vector = np.zeros((1,14))
			data_line = line.split()

			for i in range(0,14):
				example_vector[0,i] = float(data_line[i])

			data_matrix = np.vstack([data_matrix, example_vector])

			line = file.readline()
			lineCount = lineCount + 1

	print lineCount, "lines successfuly loaded."

	return data_matrix

"""
Experiment 1, Suppose we have an oracle that
tells us what records have errors but *not*
how to clean them.

Non-adaptive
"""
def doExperiment1a(uniform=False,cleaning=20):
	raw_data = loadHousingDataSet() #load the housing dataset
	
	feature_tuple = featuresObservationsSplit(raw_data,13) #get the clean result
	cleancvxsol = trainConvexLossModel(feature_tuple[0],feature_tuple[1])

	dirty_data_with_error_spec = makeMissingValues(raw_data, [1,13,0], -1, 0.1)#make it dirty
	oracle = learnMissingValuePredicate(dirty_data_with_error_spec) #learn oracle on the full data
	feature_tuple = featuresObservationsSplit(dirty_data_with_error_spec[0],13)
	cvxsol = trainConvexLossModel(feature_tuple[0],feature_tuple[1])

	importance_sampling = getImportanceSamplingDistribution(oracle[0],dirty_data_with_error_spec[0], 13, cvxsol[1], cvxsol[2],uniform)
	data = cleanNextTrainingBatch(dirty_data_with_error_spec,cleaning,importance_sampling[1],range(0,506))
	feature_tuple = featuresObservationsSplit(data[0],13) #get the clean result
	w = normalizeWeightMatrix(importance_sampling[0], int(data[2]))
	estcvxsol = trainConvexLossModel(feature_tuple[0],feature_tuple[1],'ls', w)

	return (np.linalg.norm(cleancvxsol[0]['x']-cvxsol[0]['x']), np.linalg.norm(cleancvxsol[0]['x']-estcvxsol[0]['x']))

def experiment1aResults():
	error_uniform = []
	error_importance = []
	error_dirty = []

	cleaning_range = range(2,30,2)
	averaging_trials = 25

	for cleaning in cleaning_range:

		trials_imp = []
		trials_uni = []
		trials_dirty = []

		for trial in range(0,averaging_trials):
			imp = doExperiment1a(False,cleaning)
			uni = doExperiment1a(True,cleaning)
			trials_imp.append(imp[1])
			trials_uni.append(uni[1])
			trials_dirty.append(uni[0])
			trials_dirty.append(imp[0])

		error_uniform.append(np.median(trials_uni))
		error_importance.append(np.median(trials_imp))
		error_dirty.append(np.median(trials_dirty))

	import matplotlib.pyplot as plt
	fig = plt.figure()
	uniform = plt.plot(cleaning_range,error_uniform, 'ks-', label='Uniform Sampling')
	importance = plt.plot(cleaning_range,error_importance, 'bo-', label='Importance Sampling')
	dirty = plt.plot(cleaning_range,error_dirty,'r--',label='No Cleaning')
	plt.ylabel('L_2 Error')
	plt.xlabel('# of Cleaned Examples')
	plt.grid()
	plt.title('Housing Data, Oracle, Non-Adaptive')
	plt.legend()
	plt.savefig('results/experiment1a.png')