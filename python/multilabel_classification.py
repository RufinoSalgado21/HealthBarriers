import seaborn as sns
import pandas as pd
import numpy as np
from keras import layers, Sequential
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Conv1D, Flatten, MaxPooling1D
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, hamming_loss, multilabel_confusion_matrix, precision_score, r2_score
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from python import tools

def correlations():
    df = tools.read_file('files', 'dupage_chinatown.csv')
    df_features = select_features(df)
    cols = df_features.columns.values

    #Used to return am list of the first 3 columns and all columns containing
    #the given string e.g. 'LANG' -> returns all PDLANG column names
    '''
    indexes = return_indexes_by_col_name(df_features, 'LANG')
    lst = list(df_features.columns.values[0:3])

    for i in indexes:
        lst.append(df_features.columns.values[i])
    '''
    #This set of code was dedicated for developing a heatmap plot
    #corr = df_features.corr()[['PDAGE', 'PDHSIZE', 'visit_counts']].sort_values(by='visit_counts', ascending=False)

    # Calculate correlations between all features
    corr = df_features.corr()

    #Return a list with the given feature, the highest correlation with another feature, and that feature.
    col_name = 'visit_counts'
    doc = open('correlations.txt','w')
    lst = []
    for c in cols:
        corrs = find_correlations(c, cols, corr)
        doc.write(str(corrs) + '\n')

    doc.close()

    #print(corrs)
    #heatmap = sns.heatmap(data=corr, vmin=-1, vmax=1, annot=True, cmap='BrBG')
    #plt.show()


def find_correlations(col_name, cols, corr):
    corrs = []
    m = -1
    j = 0
    vals = corr[col_name].values
    corrs.append(col_name)

    #Find max correlation and corresponding feature and append values to the output list.
    maxdex = np.where(corr.columns.values == col_name)[0]
    temp = -1.0
    for index, j in enumerate(vals):
        if index == maxdex:
            continue
        if j > temp:
            temp = j
            m = index
    corrs.append(temp)
    corrs.append(cols[m])

    #Find min correlation and append value and feature to output list.
    mindex = min(vals)
    corrs.append(mindex)
    arr = np.where(vals == mindex)
    corrs.append(cols[arr[0]][0])

    print(corrs)
    return corrs


def select_features(df):
    X = df[['PDAGE', 'visit_counts', 'PDHSIZE']]
    convert_column_to_float(X, ['PDAGE', 'PDHSIZE'])

    langs = pd.get_dummies(df['PDLANG'], prefix='PDLANG')
    zipcode = pd.get_dummies(df['PDZIP'], prefix='PDZIP')
    categories = pd.get_dummies(df['PDBIRTH'], prefix='PDBIRTH')
    mstat = pd.get_dummies(df['PDMSTAT'], prefix='Mstat')
    edu = pd.get_dummies(df['PDEDU'], prefix='Edu')
    income = pd.get_dummies(df['PDINCOME'], prefix='income')
    emp = pd.get_dummies(df['PDEMP'], prefix='Emp')
    #hospitals = pd.get_dummies(df['Nearest_hospital'], prefix='Hospital')
    encodings = [langs, zipcode, categories, mstat, edu, income, emp]
    for e in encodings:
        for col in e.columns.values:
            X[col] = e[col]
    oe = OrdinalEncoder()
    oe.fit(X)
    return X


def convert_column_to_float(X,column_list):
    for col in column_list:
        X[col] = X[col].replace('None', 0)
        X[col] = X[col].astype('float')


def select_features_visits(df):
    X = df[['PDAGE', 'PDHSIZE']]
    convert_column_to_float(X, ['PDAGE', 'PDHSIZE'])

    X['PDHSIZE'] = X['PDHSIZE'].astype('float')
    langs = pd.get_dummies(df['PDLANG'], prefix='PDLANG')
    zipcode = pd.get_dummies(df['PDZIP'], prefix='PDZIP')
    categories = pd.get_dummies(df['PDBIRTH'], prefix='PDBIRTH')
    mstat = pd.get_dummies(df['PDMSTAT'], prefix='Mstat')
    edu = pd.get_dummies(df['PDEDU'], prefix='Edu')
    income = pd.get_dummies(df['PDINCOME'], prefix='income')
    emp = pd.get_dummies(df['PDEMP'], prefix='Emp')
    #hospitals = pd.get_dummies(df['Nearest_hospital'], prefix='Hospital')
    encodings = [langs, zipcode, categories, mstat, edu, income, emp]
    for e in encodings:
        for col in e.columns.values:
            X[col] = e[col]
    oe = OrdinalEncoder()
    oe.fit(X)
    return X

def my_filter(shape, dtype=None):
    kernel = np.zeros(shape)

    lst = [0]*shape[0]
    kernel[:,0, 0] = np.array([[1,0,1]])
    #print(kernel)
    return kernel

# get the model
def get_model(n_inputs, n_outputs):
    model = Sequential()
    #c = Conv1D(32, 3, kernel_initializer=my_filter, activation='relu', input_shape=(121,1))
    c = Conv1D(32, 3, activation='relu', input_shape=(121,1))
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


'''
    Returns a keras model instance for use in predicting visit frequencies
    associated with the patient data.
'''
def get_model_visit_prediction(n_inputs):
    model = Sequential()
    model.add(Conv1D(32, 3, activation='relu', input_shape=(120, 1)))
    model.add(MaxPooling1D(pool_size=2, strides=1))
    model.add(Flatten())
    model.add(Dense(n_inputs, input_dim=n_inputs, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(n_inputs, input_dim=n_inputs, activation='relu'))
    model.add(Dense(n_inputs*2, input_dim=n_inputs, activation='relu'))
    model.add(Dense(n_inputs, input_dim=n_inputs, activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss = 'mean_squared_error', optimizer = 'adam')

    return model


'''
    Return a list with the top three labels based on the returned list from
    convert_to_prediction() which labels the top three predictions as 1, else 
    as 0.
'''
def return_predicted_labels(y_test, yhat):
    y1 = np.empty((0, 3), int)
    y2 = np.empty((0, 3), int)
    for index, i in enumerate(yhat):
        ytemp1 = [i for i in range(len(y_test[index])) if y_test[index][i] == 1]
        ytemp2 = [y_test[index][ytemp1[0]], y_test[index][ytemp1[1]], y_test[index][ytemp1[2]]]
        ytemp3 = [yhat[index][ytemp1[0]], yhat[index][ytemp1[1]], yhat[index][ytemp1[2]]]
        y1 = np.append(y1, [ytemp2], axis=0)
        y2 = np.append(y2, [ytemp3], axis=0)
    return y1, y2


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
       #print(indices)
       for ind,r in enumerate(row):
           if ind in indices:
               row[ind] = 1
           else:
               row[ind] = 0


def predict_barriers():
    df = tools.read_file('files', 'dupage_chinatown.csv')
    labels = ['barrier_1', 'barrier_2', 'barrier_3']

    #encoded_labels = pd.get_dummies(df[labels], prefix='barrier').columns
    Y = pd.get_dummies(df[labels], prefix='barrier').values
    X = select_features(df).values

    n_inputs = X.shape[1]
    n_outputs = Y.shape[1]
    #print(X[0])
    X = np.expand_dims(X, 2)
    results = evaluate_model(X, Y)
    output = open('barrier_results.txt', 'w')
    for r in results:
        output.write(str(r)+ '\n')
    output.close()


def return_indexes_by_col_name(df_features, stem):
    lst = []
    for index, col in enumerate(df_features.columns.values):
        if stem in col:
            lst.append(index)
    return lst


def predict_actions():
    df = tools.read_file('files', 'dupage_chinatown.csv')
    labels = ['action_1', 'action_2', 'action_3']

    #encoded_labels = pd.get_dummies(df[labels], prefix='actions').columns
    Y = pd.get_dummies(df[labels], prefix='actions').values
    X = select_features(df).values

    x_scalar = StandardScaler()
    x_scalar.fit(X)
    X = x_scalar.transform(X)

    n_inputs = X.shape[1]
    n_outputs = Y.shape[1]
    X = np.expand_dims(X, 2)
    results = evaluate_model(X, Y)

    output = open('action_results.txt', 'w')
    for r in results:
        output.write(str(r) + '\n')
    output.close()

def predict_visits():
    df = tools.read_file('files', 'dupage_chinatown.csv')
    labels = ['visit_counts']

    X = select_features_visits(df).values
    Y = df[labels[0]].values

    #X_reshaped = np.expand_dims(X, 2)
    #x_train, x_test, y_train, y_test = train_test_split(X_reshaped, Y, test_size=0.25, random_state=42)
    scores = []
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    for train_ix, test_ix in cv.split(X):
        # prepare data

        X_shaped = np.expand_dims(X, 2)
        n_inputs = X.shape[1]
        x_train, x_test = X_shaped[train_ix], X_shaped[test_ix]
        y_train, y_test = Y[train_ix], Y[test_ix]


        # fit model
        model = get_model_visit_prediction(n_inputs)
        model.fit(x_train, y_train, batch_size=100, verbose=False, epochs=100, validation_data=(x_test, y_test))

        # make a prediction on the test set
        yhat = model.predict(x_test)

        instance = X_shaped
        prediction = model.predict(instance)
        prediction = np.squeeze(prediction)

        for index, v in enumerate(prediction):
            prediction[index] = v.round()
        scores.append(r2_score(Y, prediction))

    print('r2 score: ',sum(scores)/len(scores))
    return scores

# evaluate a model using repeated k-fold cross-validation
def evaluate_model(X, y):
    results = list()
    accuracy = list()
    precision = list()
    recall = list()
    f1 = list()

    n_inputs, n_outputs = X.shape[1], y.shape[1]

    # define evaluation procedure
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # enumerate folds
    count = 0
    for train_ix, test_ix in cv.split(X):
        # prepare data
        #print('Round ',count)
        count += 1
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]

        # fit model
        model = get_model(n_inputs, n_outputs)
        model.fit(X_train, y_train, batch_size=100,verbose=False, epochs=50, validation_data=(X_test, y_test))

        # make a prediction on the test set
        yhat = model.predict(X_test)

        convert_to_prediction(yhat)

        confusion = multilabel_confusion_matrix(y_test, yhat)

        confusion = np.flip(confusion)

        #Calculate accuracy, precision, and recall based on confusion matrices
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for c in confusion:
            tp += c[0][0]
            tn += c[1][1]
            fp += c[1][0]
            fn += c[0][1]

        acc = (tp + tn) / np.sum(confusion)
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f = 2 * (prec * rec) / (prec + rec)

        y1, y2 = return_predicted_labels(y_test, yhat)

        hamming = hamming_loss(y2,y1)

        results.append(hamming)
        accuracy.append(acc)
        precision.append(prec)
        recall.append(rec)
        f1.append(f)
    output = []
    output.append("Hamming loss score: ")
    output.append(sum(results)/len(results))
    output.append('Accuracy score:')
    output.append(sum(accuracy)/len(accuracy))
    output.append('Precision score: ')
    output.append(sum(precision)/len(precision))
    output.append('Recall score: ')
    output.append(sum(recall) / len(recall))
    output.append('F1 score: ')
    output.append(sum(f1) / len(f1))

    return output

def main():
    '''
    correlations()
    predict_barriers()

    predict_actions()


    visits = predict_visits()

    df = tools.read_file('files', 'dupage_chinatown.csv')

    tools.create_empty_column(df, 'visits_predict', df.shape[0])
    for index, v in enumerate(visits):
        df['visits_predict'][index] = v

    cols_list = df.columns.values
    csv_list = df.values
    tools.write_to_csv(cols_list=cols_list, csv_lists=csv_list, directory='files', output_filename='dupage_chinatown.csv')
    '''
if __name__ == '__main__':
    main()
