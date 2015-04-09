#!/usr/bin/env python

import numpy as np
import random
from cvxopt import solvers, matrix, mul
from sampleml import *
from dataset import *
import matplotlib.pyplot as plt

"""
Experiment 1, Suppose we have an oracle that
tells us what records have errors but *not*
how to clean them.

Non-adaptive
"""
def doExperiment1a(uniform=False,cleaning=20, errorRate=0.1):
	raw_data = loadHousingDataSet() #load the housing dataset
	
	feature_tuple = featuresObservationsSplit(raw_data,13) #get the clean result
	cleancvxsol = trainConvexLossModel(feature_tuple[0],feature_tuple[1])

	dirty_data_with_error_spec = makeMissingValues(raw_data, [1,12, 4, 13,0], -1, errorRate)#make it dirty
	oracle = learnMissingValuePredicate(dirty_data_with_error_spec) #learn oracle on the full data
	feature_tuple = featuresObservationsSplit(dirty_data_with_error_spec[0],13)
	cvxsol = trainConvexLossModel(feature_tuple[0],feature_tuple[1])

	importance_sampling = getImportanceSamplingDistribution(oracle[0],dirty_data_with_error_spec[0], 13, cvxsol[1], cvxsol[2],uniform)
	data = cleanNextTrainingBatch(dirty_data_with_error_spec,cleaning,importance_sampling[1],range(0,506))
	feature_tuple = featuresObservationsSplit(data[0],13) #get the clean result
	w = normalizeWeightMatrix(importance_sampling[0], int(data[2]), data[4])
	estcvxsol = trainConvexLossModel(feature_tuple[0],feature_tuple[1],'ls', w)

	return (np.linalg.norm(cleancvxsol[0]['x']-cvxsol[0]['x']), np.linalg.norm(cleancvxsol[0]['x']-estcvxsol[0]['x']))

def experiment1aResults():
	error_uniform = []
	error_importance = []
	error_dirty = []

	cleaning_range = range(5,35,5)
	averaging_trials = 100

	for cleaning in cleaning_range:
		trials_imp = []
		trials_uni = []
		trials_dirty = []

		for trial in range(0,averaging_trials):
			imp = doExperiment1a(False,cleaning,0.1)
			uni = doExperiment1a(True,cleaning,0.1)
			trials_imp.append(imp[1])
			trials_uni.append(uni[1])
			trials_dirty.append(uni[0])
			trials_dirty.append(imp[0])
		
		error_uniform.append(np.median(trials_uni))
		error_importance.append(np.median(trials_imp))
		error_dirty.append(np.median(trials_dirty))
		print 'i',trials_uni, trials_imp


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

"""
Experiment 1, Suppose we have an oracle that
tells us what records have errors but *not*
how to clean them.

Adaptive
"""
def doExperiment1b(cleaning=20,batch=5, errorRate=0.1):
	raw_data = loadHousingDataSet() #load the housing dataset
	
	feature_tuple = featuresObservationsSplit(raw_data,13) #get the clean result
	cleancvxsol = trainConvexLossModel(feature_tuple[0],feature_tuple[1])

	dirty_data_with_error_spec = makeMissingValues(raw_data, [1,13,0], -1, errorRate)#make it dirty
	oracle = learnMissingValuePredicate(dirty_data_with_error_spec) #learn oracle on the full data
	feature_tuple = featuresObservationsSplit(dirty_data_with_error_spec[0],13)
	cvxsol = trainConvexLossModel(feature_tuple[0],feature_tuple[1])
	errorinitial = np.linalg.norm(cleancvxsol[0]['x']-cvxsol[0]['x'])

	for i in range(1,cleaning,batch):
		importance_sampling = getImportanceSamplingDistribution(oracle[0],dirty_data_with_error_spec[0], 13, cvxsol[1], cvxsol[2], False)
		data = cleanNextTrainingBatch(dirty_data_with_error_spec,batch,importance_sampling[1],range(0,506))
		dirty_data_with_error_spec = (data[0],data[1])
		feature_tuple = featuresObservationsSplit(dirty_data_with_error_spec[0],13) #get the clean result
		w = normalizeWeightMatrix(importance_sampling[0], int(data[2]),data[4])
		cvxsol = trainConvexLossModel(feature_tuple[0],feature_tuple[1],'ls', w)

	#do the last batch
	if cleaning%batch > 0:
		importance_sampling = getImportanceSamplingDistribution(oracle[0],dirty_data_with_error_spec[0], 13, cvxsol[1], cvxsol[2], False)
		data = cleanNextTrainingBatch(dirty_data_with_error_spec,cleaning%batch,importance_sampling[1],range(0,506))
		dirty_data_with_error_spec = (data[0],data[1])
		feature_tuple = featuresObservationsSplit(dirty_data_with_error_spec[0],13) #get the clean result
		w = normalizeWeightMatrix(importance_sampling[0], int(data[2]),data[4])
		cvxsol = trainConvexLossModel(feature_tuple[0],feature_tuple[1],'ls', w)

	return (errorinitial, np.linalg.norm(cleancvxsol[0]['x']-cvxsol[0]['x']))

def experiment1bResults():
	error_uniform = []
	error_importance_5 = []
	error_importance_10 = []
	error_importance_full = []
	error_dirty = []

	cleaning_range = range(5,30,5)
	averaging_trials = 100

	for cleaning in cleaning_range:

		trials_imp_5 = []
		trials_imp_10 = []
		trials_imp_full = []
		trials_uni = []
		trials_dirty = []

		for trial in range(0,averaging_trials):
			imp_5 = doExperiment1b(cleaning,5)
			imp_10 = doExperiment1b(cleaning,10)
			imp_full = doExperiment1b(cleaning,cleaning)

			uni = doExperiment1a(True,cleaning)
			trials_imp_5.append(imp_5[1])
			trials_imp_10.append(imp_10[1])
			trials_imp_full.append(imp_full[1])
			trials_uni.append(uni[1])
			trials_dirty.append(uni[0])
			trials_dirty.append(imp_5[0])
			trials_dirty.append(imp_10[0])
			trials_dirty.append(imp_full[0])

		error_uniform.append(np.median(trials_uni))
		error_importance_5.append(np.median(trials_imp_5))
		error_importance_10.append(np.median(trials_imp_10))
		error_importance_full.append(np.median(trials_imp_full))
		error_dirty.append(np.median(trials_dirty))

	fig = plt.figure()
	uniform = plt.plot(cleaning_range,error_uniform, 'ks-', label='Uniform Sampling')
	importance = plt.plot(cleaning_range,error_importance_5, 'b^-', label='Importance Sampling (B=5)')
	importance = plt.plot(cleaning_range,error_importance_10, 'bo-', label='Importance Sampling (B=10)')
	importance = plt.plot(cleaning_range,error_importance_full, 'bx-', label='Importance Sampling')
	dirty = plt.plot(cleaning_range,error_dirty,'r--',label='No Cleaning')
	plt.ylabel('L_2 Error')
	plt.xlabel('# of Cleaned Examples')
	plt.grid()
	plt.title('Housing Data, Oracle, Adaptive')
	plt.legend()
	plt.savefig('results/experiment1b.png')

"""
Experiment 1, Suppose we have an oracle that
tells us what records have errors but *not*
how to clean them.

Non-adaptive
"""
def doExperiment1c(uniform=False,cleaning=20, errorRate=0.1):
	raw_data = loadEEGDataSet() #load the housing dataset
	
	feature_tuple = featuresObservationsSplit(raw_data,14) #get the clean result
	cleancvxsol = trainConvexLossModel(feature_tuple[0],feature_tuple[1])

	dirty_data_with_error_spec = makeMissingValues(raw_data, [3,13,0], -1, errorRate)#make it dirty
	oracle = learnMissingValuePredicate(dirty_data_with_error_spec) #learn oracle on the full data
	feature_tuple = featuresObservationsSplit(dirty_data_with_error_spec[0],14)
	cvxsol = trainConvexLossModel(feature_tuple[0],feature_tuple[1])

	importance_sampling = getImportanceSamplingDistribution(oracle[0],dirty_data_with_error_spec[0], 14, cvxsol[1], cvxsol[2],uniform)
	data = cleanNextTrainingBatch(dirty_data_with_error_spec,cleaning,importance_sampling[1],range(0,14980))
	feature_tuple = featuresObservationsSplit(data[0],14) #get the clean result
	w = normalizeWeightMatrix(importance_sampling[0], int(data[2]),data[4])

	if not uniform:
		estcvxsol = trainConvexLossModel(feature_tuple[0],feature_tuple[1],'ls', w)
	else:
		estcvxsol = trainConvexLossModel(feature_tuple[0],feature_tuple[1],'ls', w)

	print np.linalg.norm(cleancvxsol[0]['x']-cvxsol[0]['x']), np.linalg.norm(cleancvxsol[0]['x']-estcvxsol[0]['x'])

	return (np.linalg.norm(cleancvxsol[0]['x']-cvxsol[0]['x']), np.linalg.norm(cleancvxsol[0]['x']-estcvxsol[0]['x']))

def experiment1cResults():
	error_uniform = []
	error_importance = []
	error_dirty = []

	cleaning_range = range(100,1300,100)
	averaging_trials = 2

	for cleaning in cleaning_range:

		trials_imp = []
		trials_uni = []
		trials_dirty = []

		for trial in range(0,averaging_trials):
			imp = doExperiment1c(False,cleaning,0.1)
			uni = doExperiment1c(True,cleaning,0.1)
			trials_imp.append(imp[1])
			trials_uni.append(uni[1])
			trials_dirty.append(uni[0])
			trials_dirty.append(imp[0])

		error_uniform.append(np.median(trials_uni))
		error_importance.append(np.median(trials_imp))
		error_dirty.append(np.median(trials_dirty))

	fig = plt.figure()
	uniform = plt.plot(cleaning_range,error_uniform, 'ks-', label='Uniform Sampling')
	importance = plt.plot(cleaning_range,error_importance, 'bo-', label='Importance Sampling')
	dirty = plt.plot(cleaning_range,error_dirty,'r--',label='No Cleaning')
	plt.ylabel('L_2 Error')
	plt.xlabel('# of Cleaned Examples')
	plt.grid()
	plt.title('EEG Data, Oracle, Non-Adaptive')
	plt.legend()
	plt.savefig('results/experiment1c.png')