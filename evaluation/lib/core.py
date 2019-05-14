import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import matplotlib.pyplot as plt
import os
import json
import lib.result as result
import lib.data as data
from datetime import datetime

##
# In parts of the script, we evaluate the performance of various models,
# when trained with data of a specific metric.
# During this evaluation, numeric values are implicitly converted to other data types
# The following command turns off warning messages that hint at these conversions
##
def hideWarnings():
	import warnings
	warnings.filterwarnings('ignore')

##
# This classifier is used to evaluate the predictive power of individual smells.
# Training requires a threshold parameter T for each model.
# The model is is trained using the smell strength samples of the training set only.
# Based on this information, a new sample is classified as faulty,
# if its values exceeds the bottom T % of all samples that were provided durin training.
##
class ThresholdClassifier(BaseEstimator, ClassifierMixin):

	def __init__(self, percentile=0):
		"""
		Called when initializing the classifier
		"""
		self.percentile = percentile
		self.classes_ = np.array([0, 1])
		self.n_classes_ = 2

	def fit(self, X, y=None, sample_weight=None):
		self.threshold_ = np.percentile(X, self.percentile*100)
		return self

	def _meaning(self, x):
		# returns True/False according to fitted classifier
		# notice underscore on the beginning
		if type(x) is np.ndarray:
			value = x[0]
		else:
			value = x

		return( 1 if value > self.threshold_ else 0 )

	def predict(self, X, y=None):
		try:
			getattr(self, "threshold_")
		except AttributeError:
			raise RuntimeError("You must train classifer before predicting data!")

		return([self._meaning(x) for x in X])

	def score(self, X, y=None):
		# counts number of values bigger than mean
		return(np.sum(self.predict(X) == y) / y.size)

	def _proba(self, x):
		if type(x) is np.ndarray:
			value = x[0]
		else:
			value = x

		return( [0., 1.] if value > self.threshold_ else [1., 0.] )

	def predict_proba(self, X):
		return(np.array([self._proba(x) for x in X]))

	def _func(self, x):
		if type(x) is np.ndarray:
			value = x[0]
		else:
			value = x

		return( value - self.threshold_ )

	def decision_function(self, X):
		return(np.array([self._func(x) for x in X]))

##
# The following script is used to evaluate the fault prediction performance of the individual features using specific models on a specific dataset for a specific set of testing scores.
# The model for each feature is optimized using a grid-search that includes a 10-fold cross validation.
# The optimized models are then evaluated by executing a 10 times, 10-fold cross validation test.
# In each iteration, we first split the data using a stratified split operation.
# We then apply the preprocessing operations defined for the test on the split data (standardize, resampling).
# After preprocessing, the optimized model is trained and tested with the prepared data.
# The classification results on the testing sets are gathered, and used to evaluate two different voting schemes: majority voting and advocate voting.
# The classification results on the test-data are then compared to the input-labes of the examples to calculate the required scores for the test.
##
def compareMetricCategorizationPerformances(dataset, pipeline, param_grid, model, scores, gs_scoring='f1'):
	#prepare datastructure for optimized estimators
	estimators = []

	print("Gridesarch & Cross-Validation: per-metric and voting prediction performance, training %s models, evaluated on dataset %s" %(type(model).__name__, dataset.name))

	startGridsearch = datetime.now()
	print("Gridsearch:")
	print("  start gridsearch at ", str(startGridsearch))

	# determine best estimator for each feature
	for index, val in enumerate(dataset.X[0]):
		X = dataset.X[:,index].reshape(-1,1)
		y = dataset.y

		#do gridsearch to determine best performing classifier
		gs = GridSearchCV(pipeline,
                          param_grid = param_grid,
                          scoring = gs_scoring,
                          cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True),
                          refit = gs_scoring,
                         )
		gs.fit(X, y)
		estimators.append(gs.best_estimator_)

	endGridsearch = datetime.now()
	print("  finished gridsearch at ", str(endGridsearch))
	print("  duration of gridsearch: ", str(endGridsearch - startGridsearch))
	print("  best estimators:")
	for index, estimator in enumerate(estimators):
		print("    ", index, ":", estimator)

	#prepare score names for further processing
	score_names = []
	for score in scores:
		score_names.append(score.__name__)

	#prepare data structures for evaluation data
	data = {}

	#add data structures for individual metrics
	for index, header in enumerate(dataset.h):
		estimator_data = result.initializeDataDict(name=index, label=header, scores=score_names)
		data[index] = estimator_data
	for voter in ['voting_majority', 'voting_advocate', 'voting_best']:
		voter_data = result.initializeDataDict(name=voter, label=voter, scores=score_names)
		data[voter] = voter_data


	#anouncing start of cross validatoin
	print("Cross-Validation:")
	startCrossvalidation = datetime.now()
	print("  starting cross-validation at ", str(startCrossvalidation))

	#crossvalidation
	#10 times
	for i in range(10):
		startIteration = datetime.now()
		print("    starting itaration ", i, " at ", str(startIteration))

		#initialize folds
		kfold = StratifiedKFold(n_splits=10, random_state=i, shuffle=True)
		#10 fold
		for train, test in kfold.split(dataset.X, dataset.y):
			#extract target data
			y_train = dataset.y[train]
			y_test = dataset.y[test]

			#initialize set of predictions
			preds = []

			#traind and test each classifier
			for index, val in enumerate(dataset.X[0]):
				key = dataset.h[index]
				#print(key)

				#extract training and test dataset
				X_train = dataset.X[train][:,index].reshape(-1,1)
				X_test = dataset.X[test][:,index].reshape(-1,1)

				#print("len(X_train):", len(X_train))
				#print("len(y_train):", len(y_train))
				#print("len(X_test):", len(X_test))
				#print("len(y_test):", len(y_test))

				#train estimator and predict on test data
				y_pred = estimators[i].fit(X_train, y_train).predict(X_test)

				#add prediction to prediction set
				preds.append(y_pred)
				#print(y_test, y_pred)

				#calculate scores
				for score in scores:
					data[index]['scores'][score.__name__].append(score(y_test, y_pred))

			#calculate result of majority voting
			num_predictions = len(preds)
			sum_predictions = np.sum(preds, axis=0)

			voted_predictions_majority = (sum_predictions > (num_predictions / 2)).astype(int)
			voted_predictions_advocate = (sum_predictions > 0).astype(int)
			best_predictions_faulty = np.logical_and(sum_predictions > 0, y_test).astype(int)
			voted_predictions_best = np.logical_or(sum_predictions == num_predictions, best_predictions_faulty).astype(int)

			#print("correct classes:")
			#print(y_test)

			#print("individual predictions")
			#print(preds)

			#print("meta")
			#print("num_predictions: ", num_predictions)
			#print("sum_predictions: ", sum_predictions)

			#print("voted predictions:")
			#print(voted_predictions_majority)
			#print(voted_predictions_advocate)
			#print(voted_predictions_best)

			for score in scores:
				data['voting_majority']['scores'][score.__name__].append(score(y_test, voted_predictions_majority))
				data['voting_advocate']['scores'][score.__name__].append(score(y_test, voted_predictions_advocate))
				data['voting_best']['scores'][score.__name__].append(score(y_test, voted_predictions_best))

		endIteration = datetime.now()
		print("    ending itaration ", i, " at ", str(endIteration))
		print("    duration of iteration: ", str(endIteration - startIteration))

	endCrossvalidation = datetime.now()
	print("  finished cross-validation at ", str(endCrossvalidation))
	print("  duration of crossvalidation: ", str(endCrossvalidation - startCrossvalidation))

	#prepare data structures for evaluation results
	comparision_result = result.initializeResultDict(type='classifier_comparision')
	for index, header in enumerate(dataset.h):
		estimator_result = result.initializeResultDict(type='classifier', name=index, label=header, dataset_name=dataset.name, scores=score_names) ## continue here!
		comparision_result['classifiers'][index] = estimator_result
	for voter in ['voting_majority', 'voting_advocate', 'voting_best']:
		voter_result = result.initializeResultDict(type='classifier', name=voter, label=voter, dataset_name=dataset.name, scores=score_names)
		comparision_result['classifiers'][voter] = voter_result

	print("Results:")
	for key, value in comparision_result['classifiers'].items():
		print(key, "(", value['label'],"):")
		for score in score_names:
			mean = np.mean(data[key]['scores'][score], axis=0)
			std = np.std(data[key]['scores'][score])
			comparision_result['classifiers'][key]['scores'][score]['mean'].append(mean)
			comparision_result['classifiers'][key]['scores'][score]['std'].append(std)
			comparision_result['classifiers'][key]['scores'][score]['values'] = data[key]['scores'][score]
			print("  %10s: mean=%.3f s=%.3f " % (score, mean, std))

		print() #newline

	return comparision_result, estimators

##
# The following script is used to evaluate the fault prediction performance of a provided machine learning ensemble model.
# The model is optimized using a grid-search that includes a 10-fold cross validation.
# The optimized model is then evaluated by executing a 10 times, 10-fold cross validation test.
# In each iteration, we first split the data using a stratified split operation.
# We then apply the preprocessing operations defined for the test on the split data (standardize, resampling).
# After preprocessing, the optimized model is trained and used to classify the test samples.
# The classification results on the test-data are then compared to the input-labes of the examples to calculate the required scores for the test.
##
def modelGridsearch(dataset, pipeline, param_grid, gs_scoring='f1', cv=None):
	startGridsearch = datetime.now()
	print("  start gridsearch at ", str(startGridsearch))

	if cv == None:
		cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

	#determine best estimator with gridsearch
	gs = GridSearchCV(pipeline,
                      param_grid = param_grid,
                      scoring = gs_scoring,
                      cv = cv,
                      refit = gs_scoring,
                      n_jobs=1
                     )
	gs.fit(dataset.X, dataset.y)
	best_estimator = gs.best_estimator_

	endGridsearch = datetime.now()
	print("  finished gridsearch at ", str(endGridsearch))
	print("  duration of gridsearch: ", str(endGridsearch - startGridsearch))
	print("  best estimator: ", best_estimator)

	return best_estimator

##
# The following script is used to evaluate the fault prediction performance of a provided machine learning ensemble model.
# The optimized model is then evaluated by executing a 10 times, 10-fold cross validation test.
# In each iteration, we first split the data using a stratified split operation.
# We then apply the preprocessing operations defined for the test on the split data (standardize, resampling).
# After preprocessing, the optimized model is trained and used to classify the test samples.
# The classification results on the test-data are then compared to the input-labes of the examples to calculate the required scores for the test.
##
def modelCrossVal(X, y, estimator, scores, modelName, label, datasetName=None):
	res = {}

	for score in scores:
		res[score.__name__] = {}
		res[score.__name__]['values'] = []

	#crossvalidation
	#10 times
	startCrossvalidation = datetime.now()
	print("  starting cross-validation at ", str(startCrossvalidation))
	for i in range(10):
		startIteration = datetime.now()
		print("  starting itaration ", i, " at ", str(startIteration))

		#initialize folds
		kfold = StratifiedKFold(n_splits=10, random_state=i, shuffle=True)
		#10 fold
		for train, test in kfold.split(X, y):
			#extract fold data
			X_train = X[train]
			y_train = y[train]
			X_test = X[test]
			y_test = y[test]

			#train estimator and predict on test data
			y_pred = estimator.fit(X_train, y_train).predict(X_test)

			#calculate scores
			for score in scores:
				res[score.__name__]['values'].append(score(y_test, y_pred))

		endIteration = datetime.now()
		print("  ending itaration ", i, " at ", str(endIteration))
		print("  duration of iteration: ", str(endIteration - startIteration))

	endCrossvalidation = datetime.now()
	print("  finished cross-validation at ", str(endCrossvalidation))
	print("  duration of crossvalidation: ", str(endCrossvalidation - startCrossvalidation))

	score_names = []
	for score in scores:
		score_names.append(score.__name__)

	classifier_result = result.initializeResultDict(type='classifier', name=modelName, label=label, dataset_name=datasetName, scores=score_names)

	#aggregate scores
	for score in score_names:
		#voting classifier result
		mean = np.mean(res[score]['values'], axis=0)
		std = np.std(res[score]['values'])
		classifier_result['scores'][score]['mean'].append(mean)
		classifier_result['scores'][score]['std'].append(std)
		classifier_result['scores'][score]['values'] = res[score]['values']
		print("  %s: mean=%.3f s=%.3f" % (score, mean, std))

	#return results
	return classifier_result

##
# The following script is used to evaluate the fault prediction performance of a provided machine learning ensemble model using cross-validatoin that allows to add a baseline dataset that is included in each CV iteration.
##
def modelCrossValWithBaseline(baselineDataset, cvDataset, estimator, scores, modelName, label, datasetName=None):
	res = {}

	for score in scores:
		res[score.__name__] = {}
		res[score.__name__]['values'] = []

	X = cvDataset.X
	y = cvDataset.y

	#crossvalidation
	#10 times
	startCrossvalidation = datetime.now()
	print("  starting cross-validation at ", str(startCrossvalidation))
	for i in range(10):
		startIteration = datetime.now()
		print("  starting itaration ", i, " at ", str(startIteration))

		#initialize folds
		kfold = StratifiedKFold(n_splits=10, random_state=i, shuffle=True)
		#10 fold
		for train, test in kfold.split(X, y):
			#extract fold data
			X_train = np.concatenate((baselineDataset.X,X[train]), axis=0)
			y_train = np.concatenate((baselineDataset.y,y[train]))
			X_test = X[test]
			y_test = y[test]

			#train estimator and predict on test data
			y_pred = estimator.fit(X_train, y_train).predict(X_test)

			#calculate scores
			for score in scores:
				res[score.__name__]['values'].append(score(y_test, y_pred))

		endIteration = datetime.now()
		print("  ending itaration ", i, " at ", str(endIteration))
		print("  duration of iteration: ", str(endIteration - startIteration))

	endCrossvalidation = datetime.now()
	print("  finished cross-validation at ", str(endCrossvalidation))
	print("  duration of crossvalidation: ", str(endCrossvalidation - startCrossvalidation))

	score_names = []
	for score in scores:
		score_names.append(score.__name__)

	classifier_result = result.initializeResultDict(type='classifier', name=modelName, label=label, dataset_name=datasetName, scores=score_names)

	#aggregate scores
	for score in score_names:
		#voting classifier result
		mean = np.mean(res[score]['values'], axis=0)
		std = np.std(res[score]['values'])
		classifier_result['scores'][score]['mean'].append(mean)
		classifier_result['scores'][score]['std'].append(std)
		classifier_result['scores'][score]['values'] = res[score]['values']
		print("  %s: mean=%.3f s=%.3f" % (score, mean, std))

	#return results
	return classifier_result

##
# The following script is used to evaluate the fault prediction performance of a provided machine learning ensemble model.
# The optimized model is then evaluated by executing a 10 times, 10-fold cross validation test.
# In each iteration, we first split the data using a stratified split operation.
# We then apply the preprocessing operations defined for the test on the split data (standardize, resampling).
# After preprocessing, the optimized model is trained and used to classify the test samples.
# The classification results on the test-data are then compared to the input-labes of the examples to calculate the required scores for the test.
##
def modelTrainAndTest(trainDataset, testDataset, estimator, scores, modelName, label, datasetName=None):
	res = {}

	for score in scores:
		res[score.__name__] = {}
		res[score.__name__]['values'] = []

	#Training
	y_pred = estimator.fit(trainDataset.X, trainDataset.y).predict(testDataset.X)

	#calculate scores
	for score in scores:
		res[score.__name__]['values'].append(score(testDataset.y, y_pred))

	score_names = []
	for score in scores:
		score_names.append(score.__name__)

	classifier_result = result.initializeResultDict(type='classifier', name=modelName, label=label, dataset_name=datasetName, scores=score_names)

	#aggregate scores
	for score in score_names:
		#voting classifier result
		mean = np.mean(res[score]['values'], axis=0)
		std = np.std(res[score]['values'])
		classifier_result['scores'][score]['mean'].append(mean)
		classifier_result['scores'][score]['std'].append(std)
		classifier_result['scores'][score]['values'] = res[score]['values']
		print("  %s: mean=%.3f s=%.3f" % (score, mean, std))

	#return results
	return classifier_result

##
# The following script is used to evaluate the fault prediction performance of a provided machine learning ensemble model.
# The model is optimized using a grid-search that includes a 10-fold cross validation.
# The optimized model is then evaluated by executing a 10 times, 10-fold cross validation test.
# In each iteration, we first split the data using a stratified split operation.
# We then apply the preprocessing operations defined for the test on the split data (standardize, resampling).
# After preprocessing, the optimized model is trained and used to classify the test samples.
# The classification results on the test-data are then compared to the input-labes of the examples to calculate the required scores for the test.
##
def compareMetricEnsembleCategorizationPerformance(dataset, pipeline, param_grid, model, scores, gs_scoring='f1', gridsearch_cv=None):
	modelName = type(model).__name__

	print("Gridsearch & Crossvalidation: ensemble prediction performance, training %s model, evaluated on dataset %s" %(modelName, dataset.name))

	best_estimator = modelGridsearch(dataset, pipeline, param_grid, gs_scoring='f1', cv=gridsearch_cv)
	classifier_result = modelCrossVal(dataset.X, dataset.y, best_estimator, scores, modelName, modelName, dataset.name)

	#return results
	return classifier_result, best_estimator

##
# The following script is used to evaluate the fault prediction performance of a provided machine learning ensemble model, trained on the _trainDataset_ and evaluated on the _testDataset_
##
def compareCrossDatasetEnsembleCategorizationPerformance(trainDataset, testDataset, pipeline, param_grid, model, scores, gs_scoring='f1', gridsearch_cv=None):
	modelName = type(model).__name__

	datasetName = trainDataset.name + "+" + testDataset.name
	print("Gridsearch & Transfer Learning: ensemble prediction performance, training %s model, trained on dataset %s, and evaluated on dataset %s" %(modelName, trainDataset.name, testDataset.name ))

	best_estimator = modelGridsearch(trainDataset, pipeline, param_grid, gs_scoring='f1', cv=gridsearch_cv)
	classifier_result = modelTrainAndTest(trainDataset, testDataset, best_estimator, scores, modelName, modelName, datasetName)

	#return results
	return classifier_result, best_estimator

##
# The following script allows to evaluate the prediction performance of a provided machine learning ensemble model, optimized for a combination of two datasets, and evaluated using a cross-validation strategy that combines a base dataset with a cvDataset that is used for evaluatoin.
##
def compareCrossDatasetCVEnsembleCategorizationPerformance(baseDataset, cvDataset, pipeline, param_grid, model, scores, gs_scoring='f1', gridsearch_cv=None):
	modelName = type(model).__name__

	datasetName = baseDataset.name + "+" + cvDataset.name
	print("Gridsearch & Transfer Learning CV: ensemble prediction performance, training %s model, using a base dataset of %s, and a cv dataset %s" %(modelName, baseDataset.name, cvDataset.name ))

	optimizationDataset = data.combineDatasets(baseDataset, cvDataset, baseDataset.name + "+" + cvDataset.name)

	best_estimator = modelGridsearch(optimizationDataset, pipeline, param_grid, gs_scoring='f1', cv=gridsearch_cv)
	classifier_result = modelCrossValWithBaseline(baseDataset, cvDataset, best_estimator, scores, modelName, modelName, optimizationDataset.name)

	#return results
	return classifier_result, best_estimator

##
# Calculates the importance of specific features within dataset X for the target class of y using a provided model m.
##
def calculateFeatureImportances(dataset, model):
	model.fit(dataset.X, dataset.y)#, sample_weight=dataset.w)
	importances = model.feature_importances_
	#return importances
	std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

	# Print the feature ranking
	indices = np.argsort(importances)[::-1]
	print("Feature ranking - dataset %s:" % dataset.name)
	for f in range(dataset.X.shape[1]):
		print("%d. feature %d (%s): (%f)" % (f + 1, indices[f], dataset.h[indices[f]], importances[indices[f]]))

	# Plot the feature importances of the forest
	plt.figure(figsize=(10,3))
	# plt.title("Feature importances - dataset %s" % dataset.name)
	plt.xlabel('metric')
	plt.ylabel('relative importance')
	plt.axis([0, 18, 0, 0.3])
	plt.bar(range(dataset.X.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
	plt.xticks(range(dataset.X.shape[1]), indices)
	plt.xlim([-1, dataset.X.shape[1]])
	plt.show()

## routine to apply statistic test to one data set
def applyTest(test, dataset, scaler=None):
	print("Running test \'%s\' on dataset %s and print ranked feature list" % (test.__name__, dataset.name))

	if scaler != None:
		scaler.fit(dataset.X)
		X_transformed = scaler.transform(dataset.X)
		res, pval = test(X_transformed, dataset.y)
	else:
		res, pval = test(dataset.X, dataset.y)
	#print(res)
	#print(pval)

	indices = np.argsort(res)[::-1]
	print("idx : testval - p-val     --  name")

	for index in indices:
		print ("%2d : %.2E - %.2f  --  %s" % (index, res[index], pval[index], dataset.h[index]))

	# Plot the feature importances of the forest
	plt.figure(figsize=(10,3))
	#plt.title("Test %s - dataset %s" % (test.__name__, dataset.name))
	plt.bar(range(dataset.X.shape[1]), res[indices],
           color="r", align="center")
	plt.xticks(range(dataset.X.shape[1]), indices)
	plt.xlim([-1, dataset.X.shape[1]])
	plt.xlabel('metric')
	plt.ylabel('feature importance')
	plt.show()

	#calculateFeatureImportances(enron)
	#calculateFeatureImportances(euses)
	#calculateFeatureImportances(info1)
