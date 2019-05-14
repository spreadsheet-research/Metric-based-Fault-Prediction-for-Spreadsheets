import os
import json
import csv
import io

def export(data, filename):
	print("exporting data to ", filename)

	#print("data: ", data)

	if not os.path.exists(os.path.dirname(filename)):
		try:
			os.makedirs(os.path.dirname(filename))
		except OSError as exc: # Guard against race condition
			if exc.errno != errno.EEXIST:
				raise

	with open(filename, 'w',) as fp:
		json.dump(data, fp)

	#with io.open(filename, 'w', newline='\r\n') as fp:
	#	fp.write(unicode(json.dumps(data, ensure_ascii=False)))



	print("data successfully written to ", filename)

def load(filename):
	print("importing data from ", filename)

	#print("data: ", data)

	if not os.path.exists(os.path.dirname(filename)):
		print("error: file does not exist")
		return None

	with open(filename) as data_file:
		data_loaded = json.load(data_file)

	print("data successfully imported from ", filename)
	return data_loaded

def writeCSV(data, filename, mode=None):
	print("exporting data to csv file ", filename)

	if not os.path.exists(os.path.dirname(filename)):
		try:
			os.makedirs(os.path.dirname(filename))
		except OSError as exc: # Guard against race condition
			if exc.errno != errno.EEXIST:
				raise

	with open(filename, 'wb') as fp:
		writer = csv.writer(fp, delimiter=';')

		if data['type'] == 'classifier_comparision':
			if mode == 'f1':
				classifiers = data['classifiers']
				names = []
				for key, classifier in classifiers.items():
					names.append(classifier['name'])
				writer.writerow(names)
				for i in range(0, 100):
					f1s = []
					for key, classifier in classifiers.items():
						precision = classifier['scores']['precision_score']['values'][i]
						recall = classifier['scores']['recall_score']['values'][i]
						if precision == 0 and recall == 0:
							f1 = 0
						else:
							f1 = 2 * precision * recall / (precision + recall)
						f1s.append(f1)
					writer.writerow(f1s)

			else:
				writer.writerow(('Name', 'Label', 'Precision Avg.', 'Precision Std.', 'Recall Avg.', 'Recall Std.'))
				for key in data['classifiers']:
					val = data['classifiers'][key]
					writer.writerow((val['name'], val['label'], ('%.4f' % val['scores']['precision_score']['mean'][0]).replace('.', ','),
																('%.4f' % val['scores']['precision_score']['std'][0]).replace('.', ','),
																('%.4f' % val['scores']['recall_score']['mean'][0]).replace('.', ','),
																('%.4f' % val['scores']['recall_score']['std'][0]).replace('.', ',')))

		if data['type']	== 'classifier':
			if mode == 'f1':
				writer.writerow([data['name']])
				val = data
				for i in range(0, 100):
					precision = val['scores']['precision_score']['values'][i]
					recall = val['scores']['recall_score']['values'][i]
					if precision == 0 and recall == 0:
						f1 = 0
					else:
						f1 = 2 * precision * recall / (precision + recall)
					writer.writerow([('%.4f' % f1).replace('.', ',')])
			else:
				writer.writerow(('Name', 'Label', 'Dataset', 'Precision Avg.', 'Precision Std.', 'Recall Avg.', 'Recall Std.'))
				val = data
				writer.writerow((val['name'], val['label'], val['dataset_name'], ('%.4f' % val['scores']['precision_score']['mean'][0]).replace('.', ','),
															('%.4f' % val['scores']['precision_score']['std'][0]).replace('.', ','),
															('%.4f' % val['scores']['recall_score']['mean'][0]).replace('.', ','),
															('%.4f' % val['scores']['recall_score']['std'][0]).replace('.', ',')))

		if data['type']	== 'feature_comparision':
			row = ['Name', 'Dataset']
			for index, score in enumerate(data['feature_scores']):
				row.append('Feature ' + str(index))
			writer.writerow(row)

			row = [data['name'], data['dataset_name']]
			for score in data['feature_scores']:
				row.append(('%.8f' % score).replace('.', ','))
			writer.writerow(row)

		if data['type']	== 'feature_comparison_cv':
			row = ['Name', 'Dataset', 'Index']
			for index, score in enumerate(data['feature_scores'][0]):
				row.append('Feature ' + str(index))
			writer.writerow(row)

			for index, iteration in enumerate(data['feature_scores']):
				row = [data['name'], data['dataset_name'], index]
				for score in iteration:
					row.append(('%.8f' % score).replace('.', ','))
				writer.writerow(row)

		if data['type']	== 'feature_ranking':
			row = ['Name', 'Dataset']
			for index, score in enumerate(data['feature_rankings']):
				row.append('Feature ' + str(index))
			writer.writerow(row)

			row = [data['name'], data['dataset_name']]
			for score in data['feature_rankings']:
				row.append(('%d' % score).replace('.', ','))
			writer.writerow(row)

	print("data successfully written to ", filename)

def exportEstimatorInfo(estimators, filename):
	print("exporting estimator info to file ", filename)

	if not os.path.exists(os.path.dirname(filename)):
		try:
			os.makedirs(os.path.dirname(filename))
		except OSError as exc: # Guard against race condition
			if exc.errno != errno.EEXIST:
				raise

	with open(filename, 'w') as fp:
		for index, estimator in enumerate(estimators):
			fp.write(str(index) + ":" + str(estimator) + "\n")

	print("estimator info successfully written to ", filename)

def exportDict(dict, filename):
	print("exporting dict to file ", filename)

	if not os.path.exists(os.path.dirname(filename)):
		try:
			os.makedirs(os.path.dirname(filename))
		except OSError as exc: # Guard against race condition
			if exc.errno != errno.EEXIST:
				raise

	with open(filename, 'w') as fp:
		for key in dict:
			fp.write(str(key) + ":" + str(dict[key]) + "\r\n")

	print("dict successfully written to ", filename)

##
# initializes the result dictionary for specific evaluations
##
def initializeResultDict(type, name=None, label=None, dataset_name=None, scores=None):
	result_dict = {}
	result_dict['type'] = type

	if not name == None:
		result_dict['name'] = name

	if not label == None:
		result_dict['label'] = label

	if not dataset_name == None:
		result_dict['dataset_name'] = dataset_name

	if type == 'classifier_comparision':
		result_dict['classifiers'] = {}

	elif type == 'classifier':
		result_dict['scores'] = {}
		for score in scores:
			result_dict['scores'][score] = {}
			result_dict['scores'][score]['mean'] = []
			result_dict['scores'][score]['std'] = []

	if type == 'feature_comparision':
		result_dict['feature_scores'] = []

	if type == 'feature_comparison_cv':
		result_dict['feature_scores'] = []

	if type == 'feature_ranking^':
		result_dict['feature_rankings'] = []

	return result_dict

##
# initializes the data dictionary for a specific evaluation
##
def initializeDataDict(name=None, label=None, scores=None):

	data_dict = {}

	if not name == None:
		data_dict['name'] = name

	if not label == None:
		data_dict['label'] = label

	data_dict['scores'] = {}
	for score in scores:
		data_dict['scores'][score] = []
	return data_dict

##
# combines a number of individual classifier results, whereby each result is provided as entry in the dict 'resultPaths'
##
def combineResults(resultPaths):
	comparision_result = initializeResultDict(type='classifier_comparision')

	for path in resultPaths:
		classifier_result = load(path)
		classifierName = classifier_result['name']
		comparision_result['classifiers'][classifierName] = classifier_result

	return comparision_result

##
# calculates score averages of different types within a 'classifier_comparision' result dict
# path: path to .JSON of 'classifier_comparision' result dict
# keys: keys of classifiers in the dict to include
# score_type: one of ['precision', 'recall', 'f1'] ('f1' is applied otherwise)
##
def calculateAverageScore(path, keys, score_type):
    val = load(path)
    classifierResults = val['classifiers']
    sum_score = 0
    num_score = 0

    for key in keys:
        scores = classifierResults[key]['scores']
        recall = scores['recall_score']['mean'][0]
        precision = scores['precision_score']['mean'][0]

        #add score for key
        if score_type == 'precision':
            sum_score += precision
        elif score_type == 'recall':
            sum_score += recall
        elif score_type == 'f1':
            if (precision + recall) > 0:
                sum_score += 2 * precision * recall / (precision + recall)
            # else += 0
        else: #do f1 per default
            if (precision + recall) > 0:
                sum_score += 2 * precision * recall / (precision + recall)
            # else += 0

        #increment score tally
        num_score += 1

    score = sum_score / num_score
    return score
