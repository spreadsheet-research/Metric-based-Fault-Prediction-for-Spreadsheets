Scripts to evaluate the prediction performance of ensemble models.
The results of idealized voting commitees are also part of the evaluation, but were not added to the study.

Library component is assumed to be two folders above the location of the script (e.g. '../../lib/core.py')
Can be adapted by the line "sys.path.append('../..')"

Input dataset (.csv) is assumed to be three folders above the location of the script (e.g.'../../../datasets/measured/enron.csv')
Can be adapted by the line "dataPath = '../../../datasets/measured/enron.csv'"

Output of the evaluation is assumed to be on the same folder hierarchy in the results folder (e.g. '../../results/study3')
Can be adapted by the line "exportPath = '../../results/study3'"