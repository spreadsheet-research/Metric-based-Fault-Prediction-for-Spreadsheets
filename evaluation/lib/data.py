import numpy as np
import pandas as pd
import os
import json
import collections

## Data handling function and class definitions
# DataSet class: class to contain specific dataset
class DataSet:
    def __init__(self, name):
        self.name = name
    X = None
    y = None
    w = None
    h = None
    
    def printStatistics(self):
        print("*DATASET "+self.name+"*")
        print("#total:      ", str(len(self.y)))
        print("#faulty:     ", str(sum(self.y)))
        print("#non-faulty: ", len(self.y) - sum(self.y))
        print("%faulty:     ", '{:.1%}'.format(sum(self.y) / len(self.y)))    

# removeDuplicates function: removes duplicates from data
def removeDuplicates(data_relevant):
    ##simple version
    #data_unique = np.vstack({tuple(row) for row in data_relevant})  
    
    ##elaborate version, counting number of duplicates per entry type
    d = collections.OrderedDict()
    for a in data_relevant:
        t = tuple(a)
        if t in d:
            d[t] += 1
        else:
            d[t] = 1

    result = []
    for (key, value) in d.items():
        result.append(list(key) + [value])
    B = np.asarray(result)
    return B[:,:-1], B[:,-1]

# getData function: reads data from CSV and returns DataSet
def getData(csvFile, name, removeDuplicateEntries = False):
    print("processing dataset "+name+"...")
    
    #print("Reading data of file "+csvFile)
    #d=pd.read_csv(csvFile, sep=';', dtype={'FAULTY':str}, encoding='latin-1')
    df=pd.read_csv(csvFile, sep=';', encoding='latin-1')
    #df=pd.read_csv(csvFile, sep=';', encoding='latin-1')
    
    ##extract headers
    headers = df.columns.values[4:]
    #print(headers)
    
    ##extract relevant data
    data_relevant = df.as_matrix(columns=df.columns[3:])
    
    ##converto true/false/nan to 0/1
    data_relevant[pd.isnull(data_relevant)] = 0
    #data_relevant[data_relevant != 0] = 1
    data_relevant[data_relevant == False] = 0
    data_relevant[data_relevant == True] = 1   
    #print("relevant and converted target & data:")
    #print(data_relevant)
    
    if removeDuplicateEntries:
        ##remove duplicates
        faultyDataBefore = sum(data_relevant[:,0])
        correctDataBefore = len(data_relevant) - faultyDataBefore
        data_unique, weights = removeDuplicates(data_relevant)
        faultyDataAfter = sum(data_unique[:,0])
        correctDataAfter = len(data_unique) - faultyDataAfter
        
        print("removed " + str(faultyDataBefore - faultyDataAfter) + " duplicate(s) of " + str(faultyDataBefore) + " faulty entries")
        print("removed " + str(correctDataBefore - correctDataAfter) + " duplicate(s) of " + str(correctDataBefore) + " correct entries")
        print("total entries reduced from ", str(len(data_relevant)), " to ", str(len(data_unique)))
    else:
        ##use whole dataset
        data_unique = data_relevant.astype(np.float64)
        weights = np.ones(len(data_unique))
        print("prepared " + str(len(data_relevant)) + " entries")

    ##extract data portion
    data = data_unique[:,1:]
    #print("unique data entries:")
    #print(data)
    
    ##scale data
    #data = StandardScaler().fit_transform(data)
    #print(data)
    
    ##extract target portion
    target = data_unique[:,0]
    #print("unique target entries:")
    #print(target)
    
    #data = df.as_matrix(columns=df.columns[5:])
    #print(data)
    
    #target = df.as_matrix(['FAULTY'])[:,0]
    #target[pd.isnull(target)] = 0
    #target[target != 0] = 1
    #target = target.astype(int)
    #print_full(target)
    
    dataset = DataSet(name)
    dataset.X = data
    dataset.y = target
    dataset.w = weights
    dataset.h = headers

    return dataset

# resample function: applies param 'sampler' to param 'dataset', resulting in a more balanced distribution of cases
def resample(sampler, dataset):
    X_resampled, y_resampled = sampler.fit_sample(dataset.X, dataset.y)
    #reset weights to ones
    w_resampled = np.ones(len(y_resampled))
    
    dataset_resampled = DataSet(dataset.name+" resampled")
    dataset_resampled.X = X_resampled
    dataset_resampled.y = y_resampled
    dataset_resampled.w = w_resampled
    dataset_resampled.h = dataset.h
    return dataset_resampled

def loadDataset(key, removeDuplicateEntries=False):
	return getData(DATASET[key]['csv'], DATASET[key]['name'], removeDuplicateEntries)

DEFAULT_DATASET_FILENAME = "data/datsets.json"

def exportDatasetJSON(filename = DEFAULT_DATASET_FILENAME):
	##dataset paths
	csv_minimal = 'metric_data\ENRON_errors_2017-08-01T15-41\cells_very_small.csv'
	csv_small   = 'metric_data\ENRON_errors_2017-08-01T15-41\cells_small.csv'
	csv_enron   = 'metric_data\ENRON_errors_2017-08-01T15-41\cells.csv'
	csv_euses   = 'metric_data\EUSES_mutated_2017-08-01T15-26\cells.csv'
	csv_info1   = 'metric_data\Info1_2017-08-01T15-38\cells.csv'

	csv_enron_100   = 'metric_data\ENRON_errors_2017-08-01T15-41\cells_100.csv'
	csv_enron_200   = 'metric_data\ENRON_errors_2017-08-01T15-41\cells_200.csv'
	csv_enron_500   = 'metric_data\ENRON_errors_2017-08-01T15-41\cells_500.csv'
	csv_enron_1k   = 'metric_data\ENRON_errors_2017-08-01T15-41\cells_1k.csv'
	csv_enron_2k   = 'metric_data\ENRON_errors_2017-08-01T15-41\cells_2k.csv'
	csv_enron_5k   = 'metric_data\ENRON_errors_2017-08-01T15-41\cells_5k.csv'
	csv_enron_10k   = 'metric_data\ENRON_errors_2017-08-01T15-41\cells_10k.csv'

	csv_info1_100   = 'metric_data\Info1_2017-08-01T15-38\cells_100.csv'
	csv_info1_200   = 'metric_data\Info1_2017-08-01T15-38\cells_200.csv'
	csv_info1_500   = 'metric_data\Info1_2017-08-01T15-38\cells_500.csv'
	csv_info1_1k   = 'metric_data\Info1_2017-08-01T15-38\cells_1k.csv'
	csv_info1_2k   = 'metric_data\Info1_2017-08-01T15-38\cells_2k.csv'
	csv_info1_5k   = 'metric_data\Info1_2017-08-01T15-38\cells_5k.csv'
	csv_info1_10k   = 'metric_data\Info1_2017-08-01T15-38\cells_10k.csv'
	csv_info1_20k   = 'metric_data\Info1_2017-08-01T15-38\cells_20k.csv'
	csv_info1_50k   = 'metric_data\Info1_2017-08-01T15-38\cells_50k.csv'
	
	csv_enron_journal = 'metric_data\journal_2017-12-07\enron.csv'
	csv_info1_journal = 'metric_data\journal_2017-12-07\info1.csv'
	
	dataset = {}
	dataset["mini"] = {}
	dataset["mini"]["csv"] = csv_minimal
	dataset["mini"]["name"] = "MINI"
	dataset["small"] = {}
	dataset["small"]["csv"] = csv_small
	dataset["small"]["name"] = "SMALL"
	dataset["enron"] = {}
	dataset["enron"]["csv"] = csv_enron
	dataset["enron"]["name"] = "ENRON"
	dataset["euses"] = {}
	dataset["euses"]["csv"] = csv_euses
	dataset["euses"]["name"] = "EUSES"
	dataset["info1"] = {}
	dataset["info1"]["csv"] = csv_info1
	dataset["info1"]["name"] = "INFO1"
	
	dataset["enron_100"] = {}
	dataset["enron_100"]["csv"] = csv_enron_100
	dataset["enron_100"]["name"] = "ENRON_100"
	dataset["enron_200"] = {}
	dataset["enron_200"]["csv"] = csv_enron_200
	dataset["enron_200"]["name"] = "ENRON_200"
	dataset["enron_500"] = {}
	dataset["enron_500"]["csv"] = csv_enron_500
	dataset["enron_500"]["name"] = "ENRON_500"
	dataset["enron_1k"] = {}
	dataset["enron_1k"]["csv"] = csv_enron_1k
	dataset["enron_1k"]["name"] = "ENRON_1k"
	dataset["enron_2k"] = {}
	dataset["enron_2k"]["csv"] = csv_enron_2k
	dataset["enron_2k"]["name"] = "ENRON_2k"
	dataset["enron_5k"] = {}
	dataset["enron_5k"]["csv"] = csv_enron_5k
	dataset["enron_5k"]["name"] = "ENRON_5k"
	dataset["enron_10k"] = {}
	dataset["enron_10k"]["csv"] = csv_enron_10k
	dataset["enron_10k"]["name"] = "ENRON_10k"
	
	dataset["info1_100"] = {}
	dataset["info1_100"]["csv"] = csv_info1_100
	dataset["info1_100"]["name"] = "INFO1_100"
	dataset["info1_200"] = {}
	dataset["info1_200"]["csv"] = csv_info1_200
	dataset["info1_200"]["name"] = "INFO1_200"
	dataset["info1_500"] = {}
	dataset["info1_500"]["csv"] = csv_info1_500
	dataset["info1_500"]["name"] = "INFO1_500"
	dataset["info1_1k"] = {}
	dataset["info1_1k"]["csv"] = csv_info1_1k
	dataset["info1_1k"]["name"] = "INFO1_1k"
	dataset["info1_2k"] = {}
	dataset["info1_2k"]["csv"] = csv_info1_2k
	dataset["info1_2k"]["name"] = "INFO1_2k"
	dataset["info1_5k"] = {}
	dataset["info1_5k"]["csv"] = csv_info1_5k
	dataset["info1_5k"]["name"] = "INFO1_5k"
	dataset["info1_10k"] = {}
	dataset["info1_10k"]["csv"] = csv_info1_10k
	dataset["info1_10k"]["name"] = "INFO1_10k"
	dataset["info1_20k"] = {}
	dataset["info1_20k"]["csv"] = csv_info1_20k
	dataset["info1_20k"]["name"] = "INFO1_20k"
	dataset["info1_50k"] = {}
	dataset["info1_50k"]["csv"] = csv_info1_50k
	dataset["info1_50k"]["name"] = "INFO1_50k"

	csv_enron_journal = 'metric_data\journal_2017-12-07\enron.csv'
	csv_info1_journal = 'metric_data\journal_2017-12-07\info1.csv'
	csv_euses_journal = 'metric_data\journal_2017-12-07\euses.csv'

	
	dataset["enron_journal"] = {}
	dataset["enron_journal"]["csv"] = csv_enron_journal
	dataset["enron_journal"]["name"] = "ENRON"
	dataset["info1_journal"] = {}
	dataset["info1_journal"]["csv"] = csv_info1_journal
	dataset["info1_journal"]["name"] = "INFO1"	
	dataset["euses_journal"] = {}
	dataset["euses_journal"]["csv"] = csv_euses_journal
	dataset["euses_journal"]["name"] = "EUSES"	
	
	print("writing to ", filename) 
	print(dataset)
	
	if not os.path.exists(os.path.dirname(filename)):
		try:
			os.makedirs(os.path.dirname(filename))
		except OSError as exc: # Guard against race condition
			if exc.errno != errno.EEXIST:
				raise
	
	with open(filename, 'w') as fp:
		json.dump(dataset, fp)
		
	print("written to ", filename)

def importDatasetJSON(filename = DEFAULT_DATASET_FILENAME):
	with open(filename, 'r') as fp:
		data = json.load(fp)
	return data
	
	
def testFunction(filename = DEFAULT_DATASET_FILENAME):
	print ("do nothing")
	
#import default dataset if available
from pathlib import Path
dataset_file = Path(DEFAULT_DATASET_FILENAME)

if not dataset_file.exists():
	exportDatasetJSON(DEFAULT_DATASET_FILENAME)
	
DATASET = importDatasetJSON()