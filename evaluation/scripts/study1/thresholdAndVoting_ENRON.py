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

def GSandCVforSingleMetricClassifiers(dataset, model, param_grid, exportPath, exportName, scoring=None):

    scaler = StandardScaler()
    sampler = RandomOverSampler()
    pipeline = Pipeline(steps=[('1', scaler),
                               ('2', sampler),
                               ('3', model)])

    #define scoring for cross-validatoin
    if scoring == None:
        scoring = [precision_score, recall_score]

    #gridsearch optimization & cv on dataset
    res, estimators = core.compareMetricCategorizationPerformances(dataset, pipeline, param_grid, model, scoring)

    #export result data and estimator info
    result.export(res, exportPath + "/" + exportName + ".json")
    result.writeCSV(res, exportPath + "/" + exportName + ".csv")
    result.writeCSV(res, exportPath + "/" + exportName + "_f1.csv", mode='f1')
    result.exportEstimatorInfo(estimators, exportPath + "/" + exportName + "_estimators.txt")
    
#DATASET
dataPath = '../../../datasets/measured/enron.csv'
dataName = 'ENRON'

dataset = data.getData(dataPath, dataName)
model = core.ThresholdClassifier()

param_grid = {
              "3__percentile": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
             }

exportPath = "../../results/study1"
exportName = "thresholdAndVoting_"+dataName

GSandCVforSingleMetricClassifiers(dataset, model, param_grid, exportPath, exportName)