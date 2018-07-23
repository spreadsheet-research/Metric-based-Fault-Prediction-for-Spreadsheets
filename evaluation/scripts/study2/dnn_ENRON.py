# required for evaluation on designated Linux server
import matplotlib
matplotlib.use('Agg')

import sys
sys.path.append('../..')

import lib.data as data
import lib.core as core
import lib.plot as plot
import lib.result as result
import numpy as np

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score

core.hideWarnings()

def GSandCVforEnsembleClassifier(dataset, model, param_grid, exportPath, exportName, scoring=None, sampler=None):

    scaler = StandardScaler()
    if sampler == None:
        sampler = RandomOverSampler()
    pipeline = Pipeline(steps=[('1', scaler),
                               ('2', sampler),
                               ('3', model)])

    #define scoring for cross-validatoin
    if scoring == None:
        scoring = [precision_score, recall_score]

    #gridsearch optimization & cv on dataset
    res, estimator = core.compareMetricEnsembleCategorizationPerformance(dataset, pipeline, param_grid, model, scoring)

    #export result data and estimator info
    result.export(res, exportPath + "/" + exportName + ".json")
    result.writeCSV(res, exportPath + "/" + exportName + ".csv")
    result.writeCSV(res, exportPath + "/" + exportName + "_f1.csv", mode='f1')
    result.exportEstimatorInfo([estimator.named_steps['3']], exportPath + "/" + exportName + "_estimator.txt")

#DATASET
dataPath = '../../../datasets/measured/enron.csv'
dataName = 'ENRON'

dataset = data.getData(dataPath, dataName)

from sklearn.neural_network import MLPClassifier
random_state = 42
model = MLPClassifier(max_iter=50, random_state=random_state)

#build param grid for grid search
layers = []
for i in range (1,10):
    setup = (128 for j in range(i))
    layers.append(tuple(setup))
layers
param_grid = {
				"3__hidden_layer_sizes" : layers
             }

exportPath = "../../results/study2"
exportName = "dnn_"+dataName

GSandCVforEnsembleClassifier(dataset, model, param_grid, exportPath, exportName)