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
dataPath = '../../../datasets/measured/info1.csv'
dataName = 'INFO1'

dataset = data.getData(dataPath, dataName)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()

param_grid = {
              "3__criterion" : ["gini", "entropy"],
              "3__splitter" :   ["best", "random"],
             }

exportPath = "../../results/study1"
exportName = "decisiontreeAndVoting_"+dataName

GSandCVforSingleMetricClassifiers(dataset, model, param_grid, exportPath, exportName)