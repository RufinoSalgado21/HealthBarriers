import csv
import keras
import os
import pandas as pd
import numpy as np
from keras import layers, Sequential
import matplotlib.pyplot as plt
from keras.layers import Dense
from numpy import mean, std
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.preprocessing import OrdinalEncoder
from tensorflow_core.python.keras.optimizers import SGD
from python import tools


def select_features(df):
    X = df[['PDAGE', 'Km_to_nearest_hospital']]
    for col in X.columns.values:
        X[col] = X[col].replace('None', 0)
    X['PDAGE'] = X['PDAGE'].astype('float')
    langs = pd.get_dummies(df['PDLANG'], prefix='PDLANG')
    zipcode = pd.get_dummies(df['PDZIP'], prefix='PDZIP')
    categories = pd.get_dummies(df['PDBIRTH'], prefix='PDBIRTH')
    hospitals = pd.get_dummies(df['Nearest_hospital'], prefix='Hospital')
    encodings = [langs, zipcode, categories, hospitals]
    for e in encodings:
        for col in e.columns.values:
            X[col] = e[col]
    oe = OrdinalEncoder()
    oe.fit(X)
    return X


# get the model
def get_model(n_inputs, n_outputs):
    model = Sequential()
    model.add(Dense(n_inputs * 2, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


# evaluate a model using repeated k-fold cross-validation
def evaluate_model(X, y):
    results = list()
    n_inputs, n_outputs = X.shape[1], y.shape[1]
    # define evaluation procedure
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # enumerate folds
    for train_ix, test_ix in cv.split(X):
        # prepare data
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        # define model
        model = get_model(n_inputs, n_outputs)
        # fit model
        model.fit(X_train, y_train, verbose=False, epochs=100)
        # make a prediction on the test set
        yhat = model.predict(X_test)
        convert_to_prediction(yhat)

        acc = accuracy_score(y_test, yhat)
        hamming = hamming_loss(y_test,yhat)
        print(1-hamming)

        # store result
        print('>%.3f' % acc)
        results.append(acc)
    return results

'''
    Takes the given prediction numpy arrays and convert
    each predicted instance to zeroes except for the top three
    highest scoring, labeled as ones.
'''
def convert_to_prediction(yhat):

   for index,row in enumerate(yhat):
       indices =[]
       row2 = row.tolist()
       lst = row.tolist()
       lst.sort(reverse=True)
       for m in range(3):
           indices.append(row2.index(lst[m]))

       for ind,r in enumerate(row):
           if ind in indices:
               row[ind] = 1
           else:
               row[ind] = 0

def predict_barriers():
    df = tools.read_file('datasets', 'merged_hospitals.csv')
    labels = ['R24Barrier_0', 'R24Barrier_1', 'R24Barrier_2']

    encoded_labels = pd.get_dummies(df[labels], prefix='barrier').columns
    Y = pd.get_dummies(df[labels], prefix='barrier').values
    X = select_features(df).values

    n_inputs = X.shape[1]
    n_outputs = Y.shape[1]
    results = evaluate_model(X, Y)
    print(results)

def predict_actions():
    df = tools.read_file('datasets', 'merged_hospitals.csv')
    labels = ['R24Action_0', 'R24Action_1', 'R24Action_2']

    encoded_labels = pd.get_dummies(df[labels], prefix='action').columns
    Y = pd.get_dummies(df[labels], prefix='action').values
    X = select_features(df).values

    n_inputs = X.shape[1]
    n_outputs = Y.shape[1]
    results = evaluate_model(X, Y)
    print(results)

def main():
    predict_barriers()
    predict_actions()


if __name__ == '__main__':
    main()
