#!/usr/bin/env python
# coding= UTF-8
# Non-objective-audio-classification Guillermo G. P.
# 20 January 2020
#
# This code has been extracted and adapted from sklearn
# Scikit-learn: Machine Learning in Python, Pedregosa et al.,2011.
#
# It is only used for evaluation and plotting the model.
#

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from inspect import signature



import matplotlib.pyplot as plt

def evaluate():

    # Import Results (ground truths and predictions) and Predicted Probabilities:
    results_path = 'results.npy'
    predict_filenames_path = 'predict_filenames.npy'
    predict_probs_path = 'predicted_probabilities.npy'
    filenames = np.load(predict_filenames_path)
    results = np.load(results_path)
    predicted_probs = np.load(predict_probs_path)
    
    # reduce to 1D
    predictions = results[:,1]
    ground_truths = results[:,0]
    predicted_probs = predicted_probs[:,1]
    num_files = len(predictions)
    if num_files == 0: num_files =1 #avoid division by 0.

    sum_accuracy = 0
    file = 0
    # Check the amount of correct answers:

    for pair in list(zip(predictions, ground_truths)):
        if results[file,0] == results[file,1]:
            sum_accuracy = sum_accuracy +1
        file = file + 1

    accuracy = sum_accuracy
    mean_accuracy = (accuracy/num_files) *100
    print('Num files:', num_files, ' Num correct:', accuracy, ' Mean Accuracy:', mean_accuracy, '%')


    # accuracy:
    accuracy = accuracy_score(ground_truths, predictions)
    print('Accuracy: %f' % accuracy)
    # precision
    precision = precision_score(ground_truths, predictions)
    print('Precision: %f' % precision)
    # recall:
    recall = recall_score(ground_truths, predictions)
    print('Recall: %f' % recall)
    # f1:
    f1 = f1_score(ground_truths, predictions)
    print('F1 score: %f' % f1)
    # ROC AUC
    auc = roc_auc_score(ground_truths, predicted_probs)
    print('ROC AUC: %f' % auc)
    # confusion matrix
    matrix = confusion_matrix(ground_truths, predictions)
    print('confusion matrix: ')
    print(matrix)
    
    classificationReport = classification_report(ground_truths, predictions)
    print ('Classification Report: ')
    print (classificationReport)


    ## Un-comment for obtaining a plot of Precision-Recall and ROC Curves
    """
    # Precision - Recall curve:
    average_precision = average_precision_score(ground_truths, predicted_probs)
    precision, recall, _ = metrics.precision_recall_curve(ground_truths, predicted_probs)
    step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
    plt.step(recall, precision, color='k', alpha=0.2,where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('SFX - Ambient Precision-Recall curve: AP={0:0.2f}'.format(
                                                               average_precision))

    # ROC AUC curve:
    fpr, tpr, thresholds = metrics.roc_curve(ground_truths, predicted_probs, pos_label=1)
    
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, 'b')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.show()
    """



if __name__ == '__main__':
    evaluate()
