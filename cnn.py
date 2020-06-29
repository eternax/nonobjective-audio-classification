#!/usr/bin/env python
# coding= UTF-8
# Non-objective-audio-classification Guillermo G. P.
# 20 January 2020
#
# This code has been inspired by the work “Learning Sound Event Classifiers from
# Web Audio with Noisy Labels”, arXiv preprint arXiv:1901.01189, 2019
#

import feat_extract
from feat_extract import *
import time
import argparse
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras import models
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras import utils
from tensorflow.keras import optimizers
from sklearn import preprocessing
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import os.path as op
from sklearn.model_selection import train_test_split

# Training function:
def train(args):
    if not op.exists('feat.npy') or not op.exists('label.npy'): # check if features and labels are available.
        if input('No features in the directory. Run feat_extract.py? (Y/n)').lower() in ['y', 'yes', '']:   # if not, opt. to run...
            feat_extract.main() #...the feature extraction script.
            train(args)         # and proceed to train the model.
    else:
        
        # Load features and labels.
        X = np.load('feat.npy')             # load features.
        X = np.where(np.isfinite(X), X, 0)  # if NaN values are found (due to division by 0 during feature extraction) set to 0.
        y = np.load('label.npy')            # load labels.
        y = np.ravel(y)
        # save labels as a .csv file for user check:
        np.savetxt('labels.csv', y, delimiter=',')

    X, y = shuffle(X, y)    # shuffle the data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=233)   # Split the data between train and test data.
    std_scale = preprocessing.StandardScaler().fit(X_train) # Find a proper scaling function for the data.
    X_train = std_scale.transform(X_train)                  # Apply the function to the training inputs.
    X_test = std_scale.transform(X_test)                    # Apply the function to the test inputs.
    np.savetxt('normalized_features.csv', X_train, delimiter=',')   # save the standarized features as a .csv document.
    class_count = 2;                                                # number of classes.

    # Neural Network Architecture:
    model = Sequential()
    model.add(Dense(128,  input_shape=(12,), activation='relu')) # Hidden layer 1.
    model.add(Dropout(0.2))                                      # Every 10 randomly set 2 to 0. (regularization, see report page.25)
    model.add(Dense(64, activation='relu'))                     # Hidden layer 2.
    model.add(Dense(2, activation='softmax'))                    # Output layer.
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) #set parameters of the gradient descent optimizer
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy']) # compile model use binary crossentropy loss function.

    y_train = utils.to_categorical(y_train - 1, num_classes = class_count) # format the y_train labels for the proper comparison with the prediction.
    y_test = utils.to_categorical(y_test - 1, num_classes = class_count) # format the y_test labels for the proper comparison with the prediction.
    
    start = time.time()
    model.fit(X_train, y_train, batch_size=32, epochs=500, validation_data=(X_test,y_test))  # fit the model, 32 batch size and 500 epochs.
    
    print('Training took: %d seconds' % int(time.time() - start))
    model.save(args.model)  # save model as an .h5 file

# Prediction Function:
def predict(args):
    if op.exists(args.model):
        model = models.load_model("trained_model.h5",custom_objects={'GlorotUniform': glorot_uniform()})  # load the saved model
        
        predict_feat_path = 'predict_feat.npy'
        predict_filenames = 'predict_filenames.npy'
        filenames = np.load(predict_filenames)   # load predict file features.
        X_predict = np.load(predict_feat_path)   # load  predict filenames.

        X_predict = np.where(np.isfinite(X_predict), X_predict, 0)  # if NaN values found set to 0.

        std_scale = preprocessing.StandardScaler().fit(X_predict)   # fit an appropiate standarizer for the prediction data.
        X_predict = std_scale.transform(X_predict)                  # apply the standarization to the prediction data
        predicted_labels = model.predict_classes(X_predict)         # proceed to predict the classes of the prediction (unseen) data and store
        predicted_probs = model.predict(X_predict)                  # the outputs and the probabilities into variables.
        
        for pair in list(zip(filenames, predicted_labels)):         # print each filename next to its predicted class.
            print(pair)
    
        prediction_gt_path = 'ground_truth_labels.npy'
        predict_gt_labels = np.load(prediction_gt_path)             # Load ground truth (actual) labels. (1 or 2)
        predict_gt_labels = np.array(predict_gt_labels-1)           # Convert to the same scale than the predicted data (0 or 1)
        predicted_labels = np.array(predicted_labels)
        results = list(zip(predict_gt_labels, predicted_labels))    # list the predictions and ground truth labels.
        np.save('results.npy', results)                             # save the list as an .npy file
        np.save('predicted_probabilities.npy', predicted_probs)     # save the predicted probabilities as an .npy file

    elif input('Model not found. Train network first? (Y/n)').lower() in ['y', 'yes', '']:  # if prediction is ran and model not found, opt. to train the network.
        train()             #train
        predict(args)       # and predict.

def main(args):
    if args.train: train(args)
    elif args.predict: predict(args)

if __name__ == '__main__':                                      #parse arguments are explained in the readme.txt file
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-t', '--train',             action='store_true',                           help='train neural network')
    parser.add_argument('-m', '--model',             metavar='path',     default='trained_model.h5',help='use this model')
    parser.add_argument('-p', '--predict',           action='store_true',                           help='predict files')
    args = parser.parse_args()
    main(args)
