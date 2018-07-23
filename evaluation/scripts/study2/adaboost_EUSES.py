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

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

random_state = 42
baseClassifier = DecisionTreeClassifier(random_state = random_state, max_features = "auto",max_depth = None)
model = AdaBoostClassifier(base_estimator = baseClassifier, random_state=random_state)

#build param grid for grid search
param_grid = {"3__base_estimator__criterion" : ["gini", "entropy"],
              "3__base_estimator__splitter" :   ["best", "random"],
              "3__n_estimators": [1, 5, 10, 25, 50]
             }

exportPath = "../../results/study2"
exportName = "adaboost_"+dataName

GSandCVforEnsembleClassifier(dataset, model, param_grid, exportPath, exportName)