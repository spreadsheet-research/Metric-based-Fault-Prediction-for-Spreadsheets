This folder contains plots that describe the results of additional experiments that were conducted during our evaluation.

comparison_NIER:    			Compares the performance of ensemble classifiers trained on all 64 metrics, with classifiers trained on 17 smell detection metrics that were presented in previous work (NIER), trained and evaluated on the Enron Errors dataset.

optimization_<DATASET>: 		Compares the performance of ensemble classifiers that were parameter-optimized for the F1-score to ones that were parameter-optimized for precision, trained and evaluated on the <DATASET>={Enron Errors, INFO1, EUSES} dataset.

oversampling_<DATASET>: 		Compares the performance of ensemble classifiers that were trained using ADASYN and SMOTE oversampling routhines to ones that were trained using random oversampling (our default approach), trained and evaluated on the <DATASET>={Enron Errors, INFO1, EUSES} dataset.

singleMetric_<CLASSIFIER>_<DATASET>:	Compares the performance of <CLASSIFIER>={decision tree, logistic regression, threshold} classifiers that use only a single metric for classification and were trained and evaluated on the <DATASET>={Enron Errors, INFO1, EUSES} dataset.