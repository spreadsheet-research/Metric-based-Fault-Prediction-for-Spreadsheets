# required for evaluation on Pantora
import matplotlib
matplotlib.use('Agg')

import sys
# Add the ptdraft folder path to the sys.path list
sys.path.append('/home/pkoch/FaultPrediction')

import lib.data as data
import lib.core as core
import lib.plot as plot
import lib.result as result
import numpy as np

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import precision_score, recall_score
from datetime import datetime

core.hideWarnings()

def evaluateFeatureScores(dataset, exportPath, exportName):

	model = RandomForestClassifier(random_state = 42)

	#build param grid for grid search
	param_grid = {"3__criterion" : ["gini", "entropy"],
                  "3__n_estimators": [1, 5, 10, 25, 50]
                 }

	scaler = StandardScaler()
	sampler = RandomOverSampler()
	pipeline = Pipeline(steps=[('1', scaler),
                               ('2', sampler),
                               ('3', model)])

	gs = GridSearchCV(pipeline,
                      param_grid = param_grid,
                      scoring = 'f1',
                      cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True),
                      refit = 'f1',
                      n_jobs=1
                     )
	print("  starting gridsearch at ", str(datetime.now()))
	gs.fit(dataset.X, dataset.y)
	best_estimator = gs.best_estimator_

	scores = [precision_score, recall_score]

	res = {}
	feature_importances = []

	for score in scores:
		res[score.__name__] = {}
		res[score.__name__]['values'] = []

	X = dataset.X
	y = dataset.y

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
			y_pred = best_estimator.fit(X_train, y_train).predict(X_test)

			#calculate scores
			for score in scores:
				res[score.__name__]['values'].append(score(y_test, y_pred))

			#add feature importances to list
			feature_importances.append(best_estimator.steps[2][1].feature_importances_.tolist())

		endIteration = datetime.now()
		print("  ending itaration ", i, " at ", str(endIteration))
		print("  duration of iteration: ", str(endIteration - startIteration))

	endCrossvalidation = datetime.now()
	print("  finished cross-validation at ", str(endCrossvalidation))
	print("  duration of crossvalidation: ", str(endCrossvalidation - startCrossvalidation))

	score_names = []
	for score in scores:
		score_names.append(score.__name__)

	classifier_result = result.initializeResultDict(type='classifier', name='randomForest', label='randomForest_CV', dataset_name=dataset.name, scores=score_names)

	#aggregate scores
	for score in score_names:
		#voting classifier result
		mean = np.mean(res[score]['values'], axis=0)
		std = np.std(res[score]['values'])
		classifier_result['scores'][score]['mean'].append(mean)
		classifier_result['scores'][score]['std'].append(std)
		classifier_result['scores'][score]['values'] = res[score]['values']
		print("  %s: mean=%.3f s=%.3f" % (score, mean, std))

	res = result.initializeResultDict(type='feature_comparison_cv', name='feature importances', dataset_name=dataset.name)
	res['feature_scores'] = feature_importances

	#export result data and estimator info
	result.export(classifier_result, exportPath + "/" + exportName + "_classification" + ".json")
	result.writeCSV(classifier_result, exportPath + "/" + exportName + "_classification" + ".csv")
	result.exportEstimatorInfo([best_estimator], exportPath + "/" + exportName + "_estimator.txt")

	result.export(res, exportPath + "/" + exportName + "_featureImportances" + ".json")
	result.writeCSV(res, exportPath + "/" + exportName + "_featureImportances" + ".csv")

#ENRON
enronPath = '../../../metric_data/journal_2017-12-07/enron.csv'
enronName = 'ENRON'
enronDataset = data.getData(enronPath, enronName)

#INFO1
info1Path = '../../../metric_data/journal_2017-12-07/info1.csv'
info1Name = 'INFO1'
info1Dataset = data.getData(info1Path, info1Name)

#EUSES
#eusesPath = '../../../metric_data/journal_2017-12-07/euses.csv'
#eusesName = 'EUSES'
#eusesDataset = data.getData(eusesPath, eusesName)

dataName = enronName + "+" + info1Name
dataset = data.combineDatasets(enronDataset, info1Dataset, dataName)

exportPath = "../../../results/2019-04/featureImportance"
exportName = "featureImportance_randomForest_"+dataName

evaluateFeatureScores(dataset, exportPath, exportName)
