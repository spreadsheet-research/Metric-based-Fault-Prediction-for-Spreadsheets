# Fault prediction with multiple metrics (Study 3)

Results of the third study are presented in the same form as in Study 2. *knn* files contain the evaluation results for k-NN only, whereas *knn_sfs* -- for kNN with feature selection. 

    + *.json files contain all measured precision and recall values.
    + *_f1.csv files contain the corresponding F1 values. The first line comprises the name of a classifier and all other lines the values using comma as the decimal separator.
    + *.csv files are semicolon separated value files containing a table presenting averages of the precision, recall and F1 measure.
    + *_features.txt contains the best achieved F1 score and the corresponding set of features computed by the recursive algortithm.