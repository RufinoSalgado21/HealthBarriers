import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import layers, Sequential
from keras.layers import Dense, Dropout, Conv1D, Flatten, MaxPooling1D
from sklearn.metrics import accuracy_score, hamming_loss, multilabel_confusion_matrix, precision_score, r2_score, \
    confusion_matrix
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MultiLabelBinarizer
from python import tools, preprocess_dupage_chinatown, dupage_preprocessing, chinatown_preprocessing, visit_label_classification


def calculate_positive_negatives(classes, class_label, num_of_patients, y_test, yhat,
                                 yhat_labels, ytest_labels):

    #print('labels',labels)
    class_index = np.where(classes == class_label)
    trueP = 0
    falseP = 0
    trueN = 0
    falseN = 0
    # For every instance
    misclassification = []
    for index in num_of_patients:
        if y_test[:, class_index][index] == 1:
            if yhat[:, class_index][index] == 1:
                trueP += 1
            else:
                falseN += 1
                misclassification.append([index, ytest_labels[index], yhat_labels[index]])
        else:
            if yhat[:, class_index][index] == 0:
                trueN += 1
            else:
                falseP += 1
    confusion = np.array([[trueN, falseP], [falseN, trueP]])
    '''
    positives = trueP + falseN
    if positives != 0:
        output.write('\nCurrent Label:' + str(classes[class_index]) + '\n')
        output.write('Expected:' + str(positives) + '\n')
        output.write('Correct Classified: ' + str(trueP) + '\n')
        output.write('Misclassified:' + str(falseN) + ' : \n')
        output.write('\tExpected Labels\t\t\tPredicted Labels\n')
        for m in misclassification:
            output.write(str(m) + '\n')
    '''
    return confusion

def select_features(df):
    X = df[['PDAGE','visit_counts', 'PDHSIZE']]
    tools.convert_column_to_float(X, ['PDAGE', 'PDHSIZE'])
    langs = pd.get_dummies(df['PDLANG'], prefix='PDLANG')
    zipcode = pd.get_dummies(df['PDZIP'], prefix='PDZIP')
    categories = pd.get_dummies(df['PDBIRTH'], prefix='PDBIRTH')
    mstat = pd.get_dummies(df['PDMSTAT'], prefix='Mstat')
    edu = pd.get_dummies(df['PDEDU'], prefix='Edu')
    income = pd.get_dummies(df['PDINCOME'], prefix='income')
    emp = pd.get_dummies(df['PDEMP'], prefix='Emp')
    #visits = pd.get_dummies(df['visit_ranges'], prefix='Visits')
    encodings = [langs, zipcode, categories, mstat, edu, income, emp]
    for e in encodings:
        for col in e.columns.values:
            X[col] = e[col]
    csv_path = os.environ['PYTHONPATH'] + os.path.sep + 'files' + os.path.sep + 'X.csv'
    print(X)
    datax = pd.DataFrame(X).to_csv(path_or_buf=csv_path)
    return X

# get the model
def get_model(n_inputs, n_outputs):
    model = Sequential()
    c = Conv1D(32, 3, activation='relu', input_shape=(n_inputs,1))
    model.add(c)
    model.add(MaxPooling1D(pool_size=2, strides=1))
    model.add(Flatten())
    model.add(Dense(n_inputs, input_dim=n_inputs, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(n_inputs, input_dim=n_inputs, activation='relu'))
    model.add(Dense(n_inputs*2, input_dim=n_inputs, activation='relu'))
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def eval_barriers(directory, filename):
    print('barriers')
    df = tools.read_file(directory,filename)

    labels = ['barrier_1', 'barrier_2', 'barrier_3']

    Y, mlb = tools.select_multilabels(df, labels)
    X = select_features(df).values

    '''
    x_scalar = StandardScaler()
    X[:, 0:3] = x_scalar.fit_transform(X[:, 0:3])
    '''
    X = np.expand_dims(X, 2)
    evaluate_model(X, Y, mlb.classes_,output_filename='barrier_misclassifications.csv', metrics_filename='barriermetrics.csv')

def eval_actions(directory, filename):
    print('actions')
    df = tools.read_file(directory, filename)

    labels = ['action_1', 'action_2', 'action_3']

    Y, mlb = tools.select_multilabels(df, labels)
    X = select_features(df).values

    csv_path2 = os.environ['PYTHONPATH'] + os.path.sep + 'files' + os.path.sep + 'y.csv'
    datay = pd.DataFrame(Y, columns=mlb.classes_).to_csv(path_or_buf=csv_path2)
    '''
    #Scaling should only be applied to non-encoded values
    x_scalar = StandardScaler()
    X[:, 0:3] = x_scalar.fit_transform(X[:, 0:3])
    '''
    X = np.expand_dims(X, 2)
    evaluate_model(X, Y, mlb.classes_, 'action_misclassifications.csv', 'actionmetrics.csv')

# evaluate a model using repeated k-fold cross-validation
def evaluate_model(X, y, classes,output_filename, metrics_filename):
    n_inputs, n_outputs = X.shape[1], y.shape[1]
    model = get_model(n_inputs, n_outputs)

    # define evaluation procedure
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=101)
    measures = {}
    for i in classes:
        measures[i] = [0]*4

    count = -1
    # enumerate folds
    for train_ix, test_ix in cv.split(X):
        count += 1

        # prepare data
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]

        # fit model
        model.fit(X_train, y_train, batch_size=100,verbose=False, epochs=50, validation_data=(X_test, y_test))

        # make a prediction on the test set
        yhat = model.predict(X_test)
        tools.convert_to_binary(yhat, y_test)
        ytest_labels = tools.return_predicted_labels(y_test, classes)
        yhat_labels = tools.return_predicted_labels(yhat, classes)

        num_of_classes = range(len(yhat[0]))
        num_of_patients = range(len(yhat[:, 0]))

        all_labels = []
        all_true = []
        all_pred = []
        for i in range(len(ytest_labels)):
            all_labels.extend(ytest_labels[i])
            all_true.extend(ytest_labels[i])
            all_labels.extend(yhat_labels[i])
            all_pred.extend(yhat_labels[i])
        all_labels = np.unique(all_labels).tolist()

        con = confusion_matrix(all_true, all_pred, labels=all_labels)

        expected = [all_true.count(i) for i in all_labels]
        predicted = [all_pred.count(i) for i in all_labels]
        con_frame = pd.DataFrame(con, columns=all_labels)
        con_frame['classes'] = all_labels
        con_frame['expected'] = expected
        con_frame['predicted'] = predicted
        con_frame.set_index(['classes', 'expected', 'predicted'], inplace=True)

        #Determine cases for each class misclassified and as what
        csv_path = os.environ['PYTHONPATH'] + os.path.sep + 'files' + os.path.sep + output_filename
        data = con_frame.to_csv(path_or_buf=csv_path)

        #Calculate and store accuracy, precision, recall, and F1 scores for this fold to dictionary.
        #Output the average values from all folds as a csv file
        for lab in all_labels:
            confusion = calculate_positive_negatives(classes, lab, num_of_patients,
                                                 y_test, yhat, yhat_labels, ytest_labels)

            results = tools.calculate_confusion_matrix_measures(confusion)
            measures[lab] = np.divide(np.add(measures[lab],results), 2)

    measures_df = pd.DataFrame(measures)#, columns=['Class', 'Precision', 'Recall', 'F1 Score'])
    measures_df['Metrics'] = ['Accuracy','Precision','Recall','F1 Score']
    measures_df.set_index(['Metrics'],inplace=True)
    measure_path = os.environ['PYTHONPATH'] + os.path.sep + 'files' + os.path.sep + metrics_filename
    measures_df.to_csv(path_or_buf=measure_path)


def main():

    print('PREDICTING BARRIERS...')
    eval_barriers('files', 'dupage_chinatown.csv')
    print('PREDICTING ACTIONS..')
    eval_actions('files', 'dupage_chinatown.csv')

if __name__ == '__main__':
    main()
