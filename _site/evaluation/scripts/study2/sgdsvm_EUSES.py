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
    result.exportEstimatorInfo([estimator], exportPath + "/" + exportName + "_estimator.txt")

#DATASET
dataPath = '../../../datasets/measured/euses.csv'
dataName = 'EUSES'

dataset = data.getData(dataPath, dataName)

from sklearn.linear_model import SGDClassifier

model = SGDClassifier()

#build param grid for grid search
param_grid = {'3__alpha': 10.0**-np.arange(1,7)}

exportPath = "../../results/study2"
exportName = "sgdsvm_"+dataName

GSandCVforEnsembleClassifier(dataset, model, param_grid, exportPath, exportName)