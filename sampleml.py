#!/usr/bin/env python

import numpy as np
import random
from cvxopt import solvers, matrix, mul

"""
Introduce missing values with a given place holder value.
The function also returns an dict that we call
an errorspec to keep track of how we corrupted the data.
"""
def makeMissingValues(data_matrix, cols, placeholder, prob=0.1):
	size = data_matrix.shape
	errorspec = {}

	lineCount = 0
	for i in range(0,size[0]):
		if random.random() < prob:
			col = random.choice(cols)
			
			#error type, col, corrupted val, original val
			errorspec[i] = [('missing', col, placeholder, data_matrix[i,col])]

			data_matrix[i,col] = placeholder
			lineCount = lineCount + 1

	print lineCount, "Examples corrupted"

	return (data_matrix, errorspec)

"""
Learn missing value predicate from sample.
This takes as inpput the data error tuple which 
is the data matrix and an errorspec dict.

Returns a result tuple List[(col, predicate, estimated change)]
a cost value that tells you how much you spent on cleaning
an efficiency value that tells you tells you how much of
the "cleaned data" was actually dirty.
"""
def learnMissingValuePredicate(data_error_tuple):
	data_matrix = data_error_tuple[0]
	errorspec = data_error_tuple[1]
	cleaned_tuples = 0
	
	predicates = {}

	#for all records
	for k in errorspec:

		#for all errors
		error_list = errorspec[k]
		for error in error_list:

			if error[0] == 'missing':

				#group
				predicate_key = (error[1],error[2])

				if predicate_key not in predicates:
					predicates[predicate_key] = [] 
				predicates[predicate_key].append(error[3] - error[2])
				cleaned_tuples = cleaned_tuples + 1

	result_tuple = []
	#for all missing value errors
	for k in predicates:
		result_tuple.append(
							(   k[0],
							 	lambda data, col: (data[col] == k[1]),
							 		np.mean(predicates[(k[0],
							 					 	k[1])]
							 			)
								)
						    )

	print cleaned_tuples, "examples with missing vals learned predicate"

	return (result_tuple, 
			np.shape(data_matrix)[0], 
			(cleaned_tuples + 0.0)/np.shape(data_matrix)[0])

"""
This function splits the feature vectors and the observations
using numpy's slicing feature
"""
def featuresObservationsSplit(data_matrix, obscol):
	size = data_matrix.shape
	feature_indices = set(range(0,size[1])).difference(set([obscol]))
	return (data_matrix[:,list(feature_indices)], data_matrix[:,obscol])

"""
Trains a convex loss model, takes in a feature matrix, an observation_matrix, problem type
and weights (as a matrix).
Returns a cvxopt solution, a loss, and a gradient at each point
"""
def trainConvexLossModel(feature_matrix, observation_matrix, opttype='ls', weights=None):
	
	N = feature_matrix.shape[0]
	p = feature_matrix.shape[1]

	if weights == None:
		weights = np.eye(N)

	sol = None
	gradient = np.zeros((N,p+1))
	loss = np.zeros((N,1))

	#solves least squares problems
	if opttype == 'ls':
		weighted_feature_matrix = np.dot(np.transpose(feature_matrix), weights)
		kernel = matrix(
							np.dot(weighted_feature_matrix,
									feature_matrix)
					   )
		obs = matrix(
							-np.dot(weighted_feature_matrix,
									observation_matrix)
					)
		
		#cvx do magic!
		sol = solvers.qp(kernel, obs)

		for i in range(0,N):
			residual = (np.dot(np.transpose(sol['x']),feature_matrix[i,:]) - observation_matrix[i])
			gradient[i,0:p] = residual*np.transpose(sol['x'])
			gradient[i,p] = residual*-1
			loss[i] = residual ** 2

	return (sol,loss,gradient)

"""
This calculates the importance sampling distribution.
Only works for missing vals now
"""
def getImportanceSamplingDistribution(cleaning_result_tuple,
									  data_matrix,
									  obscol,
									  loss,
									  gradient,
									  force_uniform=False):
	N = data_matrix.shape[0]
	p = data_matrix.shape[1]
	num_errors = 0.0

	#stupid hack for now
	feature_indices = list(set(range(0,p)).difference(set([obscol])))

	importance_weights = loss
	for i in range(0,N):
		estimated_feature_change = np.zeros((p-2,1))
		estimated_obs_change = 0
		
		erroneous = 0

		for j in cleaning_result_tuple:
			if j[1](data_matrix[i,:],j[0]):
				if j[0] != obscol:
					estimated_feature_change[feature_indices.index(j[0])] = j[2]
				else:
					estimated_obs_change = j[2]

				erroneous = 1.0

		num_errors = num_errors + erroneous
		importance_weights[i] = erroneous*abs(importance_weights[i] + np.dot(gradient[i,0:(p-2)],estimated_feature_change) + gradient[i,(p-1)]*estimated_obs_change)

	normalization = np.sum(importance_weights)
	standard_normalization = 1.0/max(num_errors,1.0)

	weight_matrix = np.zeros((N,N))
	sampling_probs = np.zeros((N,1))
	for i in range(0,N):
		if importance_weights[i] != 0 and not force_uniform:
			weight_matrix[i,i] = importance_weights[i] / normalization
			sampling_probs[i] = (importance_weights[i] / normalization)

		if importance_weights[i] != 0 and force_uniform:
			weight_matrix[i,i] = 1.0
			sampling_probs[i] = 1.0/num_errors

		if num_errors == 0:
			weight_matrix[i,i] = 1.0
			sampling_probs[i] = 1.0/N


	return (weight_matrix, sampling_probs)

"""
This gets the next batch given an importance sampling distribution
"""
def getNextTrainingBatch(data_matrix_es_tuple,k,sampling_probs,population):
	selectedIndex = np.random.choice(population, k, False, np.array(sampling_probs)[:,0])
	errorspec = {}
	for i in selectedIndex:
		if i in data_matrix_es_tuple[1]:
			errorspec[i] = data_matrix_es_tuple[1][i]
	return (data_matrix_es_tuple[0][selectedIndex,:], errorspec)

"""
This cleans the next training batch
"""
def cleanNextTrainingBatch(data_matrix_es_tuple,k,sampling_probs,population):
	selectedIndex = np.random.choice(population, k, False, np.array(sampling_probs)[:,0])
	
	cost = 0.0
	actual_cleaned = 0.0

	errorspec = data_matrix_es_tuple[1]
	for i in selectedIndex:
		if i in data_matrix_es_tuple[1]:			
			for j in data_matrix_es_tuple[1][i]:
				if j[0] == 'missing':
					data_matrix_es_tuple[0][i,j[1]] = j[3]

			del errorspec[i]

			actual_cleaned = actual_cleaned + 1

		cost = cost + 1

	print cost,'examples looked at.',actual_cleaned, " examples cleaned"

	return (data_matrix_es_tuple[0], errorspec, cost, actual_cleaned)	

"""
This normalizes the weight matix for re-estimation after cleaning
"""
def normalizeWeightMatrix(weight_matrix, cleaned):
	N = weight_matrix.shape[0]
	K = len(np.flatnonzero(weight_matrix)) + 0.0
	for i in range(0,N):
		if weight_matrix[i,i] == 0:
			weight_matrix[i,i] = 1
		else:
			weight_matrix[i,i] = K/cleaned*weight_matrix[i,i]
			
	return weight_matrix