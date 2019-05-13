# required for evaluation on designated Linux server
import matplotlib
matplotlib.use('Agg')

import sys
sys.path.append('../..')

import lib.data as data
import lib.core as core
import lib.result as result
import numpy as np

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import precision_score, recall_score

core.hideWarnings()

def SFSforClassifier(dataset, model, exportPath, exportName, modelName, scoring=None, sampler=None):

    scaler = StandardScaler()
    if sampler == None:
        sampler = RandomOverSampler()
    pipeline = Pipeline(steps=[('1', scaler),
                               ('2', sampler),
                               ('3', model)])

    random_state = 42
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
							   
    sfs = SFS(estimator=pipeline, 
           k_features="best",
           forward=False, 
           floating=True, 
           scoring='f1',
           cv=kfold,
           n_jobs=-1)
	
	
    sfs.fit(dataset.X, dataset.y)
	
    res = {}
    res['features'] = sfs.k_feature_idx_
    res['score'] = sfs.k_score_
	    
    #export sfs data and estimator info
    result.exportDict(res, exportPath + "/" + exportName + ".txt")
    X_sfs = sfs.transform(dataset.X)
	
    #define scoring for cross-validatoin
    scores = [precision_score, recall_score]
	
    #gridsearch optimization & cv on dataset
    res = core.modelCrossVal(X_sfs, dataset.y, pipeline, scores, modelName, modelName, datasetName=dataset.name)
		
    #export result data and estimator info	
    result.export(res, exportPath + "/" + exportName + ".json")
    result.writeCSV(res, exportPath + "/" + exportName + ".csv")
    result.writeCSV(res, exportPath + "/" + exportName + "_f1.csv", mode='f1')
	
#DATASET
dataPath = '../../../datasets/measured/enron.csv'
dataName = 'ENRON'

dataset = data.getData(dataPath, dataName)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3, weights='distance')

exportPath = "../../results/study3"
exportName = "knn_sfs_"+dataName
modelName = type(model).__name__

SFSforClassifier(dataset, model, exportPath, exportName, modelName)