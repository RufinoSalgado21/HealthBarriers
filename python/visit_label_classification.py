import os

import pandas as pd
from sklearn import svm, metrics
from sklearn.metrics import r2_score, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import RepeatedKFold, train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from python import tools
import numpy as np
import matplotlib.pyplot as plt

def encode_input(age, householdsize, language, zipcode, birth_country, marital_status, education_level, income_level, employment_status):
    features = [ ]
    age = float(age)
    hsize = householdsize
    lang = return_lang_encoded(language)
    zip = return_zip_encoded(zipcode)
    country = return_birth_encoded(birth_country)
    mstat = return_mstat_encoded(marital_status)
    edu = return_edu_encoded(education_level)
    income = return_income_encoded(income_level)
    employment = return_empstat_encoded(employment_status)
    x = [age, hsize]
    x.extend(lang)
    x.extend(zip)
    x.extend(country)
    x.extend(mstat)
    x.extend(edu)
    x.extend(income)
    x.extend(employment)

    return x

def return_empstat_encoded(employment_status):
    status = ['housewife', 'working', 'retired', 'unemployed', 'other', 'student']

    arr = [0, 0, 0, 0, 0, 0]
    if employment_status is None:
        arr[4] = 1
        return arr

    for index, s in enumerate(status):
        if employment_status == s:
            arr[index] = 1
        else:
            arr[index] = 0

    return arr

def return_income_encoded(income_level):
    levels = ['income_1.0', 'income_2.0' 'income_3.0' 
              'income_4.0' 'income_5.0' 'income_6.0'
            'income_98.0' ]
    arr = [0, 0, 0, 0, 0, 0, 0,0]
    if income_level is None:
        arr[6] = 1
    elif income_level < 10000:
        arr[0] = 1
    elif income_level < 20000:
        arr[1] = 1
    elif income_level < 30000:
        arr[2] = 1
    elif income_level < 40000:
        arr[3] = 1
    elif income_level < 50000:
        arr[4] = 1
    elif income_level >= 50000:
        arr[5] = 1

    return arr


def return_edu_encoded(education):
    levels = ['8th grade or less', 'some high school', 'high school diploma',
            'some college/vocational', 'associate degree',
            'college graduate', 'graduate/professional degree', 'choose not to answer']
    if education is None:
        education = 'choose not to answer'

    arr = [0,0,0,0,0,0,0,0]
    for index, lev in enumerate(levels):
        if education.lower() == lev:
            arr[index] = 1
        else:
            arr[index] = 0

    if 1 not in np.unique(arr):
        arr[7] = 1
    return arr

def return_mstat_encoded(marital_status):
    status = ['single', 'married/living as married', 'divorced/separated',
         'widowed', 'choose not to answer']

    if marital_status is None:
        marital_status = 'choose not to answer'
    #arr = [1 if marital_status.lower() in c else 0 for c in status]
    arr = [0,0,0,0,0]
    for index, m in enumerate(status):
        if marital_status.lower() == m:
            arr[index] = 1
        else:
            arr[index] = 0

    if 1 not in np.unique(arr):
        arr[4] = 1
    return arr



def return_birth_encoded(birth_country):
    countries = ['afghanistan',
         'albania', 'angola', 'argentina', 'bangladesh', 'bolivia',
         'chile', 'china', 'colombia', 'cuba', 'germany',
         'ecuador', 'greece', 'guatemala', 'honduras', 'india',
         'iran', 'italy', 'jamaica', 'jordan', 'lebanon',
         'lithuania', 'mexico', 'none', 'peru', 'philippines',
         'poland', 'puerto rico', 'russia', 'sudan', 'el salvador',
         'united kingdom', 'united states', 'vietnam']
    arr = [0]*len(countries)
    if birth_country is None:
        birth_country = 'None'
    if birth_country.lower() is 'us':
        birth_country = 'united states'

    for index, c in enumerate(countries):
        if birth_country.lower() in c:
            arr[index] = 1
        else:
            arr[index] = 0
    if 1 not in np.unique(arr):
        arr[np.where(countries == 'none')] = 1
    return arr

def return_zip_encoded(zipcode):
    if zipcode is None:
        zipcode = 'None'
    zip = str(zipcode)
    codes = ['PDZIP_60101', 'PDZIP_60103', 'PDZIP_60106', 'PDZIP_60108',
             'PDZIP_60126', 'PDZIP_60133', 'PDZIP_60134', 'PDZIP_60137', 'PDZIP_60139',
             'PDZIP_60143', 'PDZIP_60148', 'PDZIP_60166', 'PDZIP_60172', 'PDZIP_60181',
             'PDZIP_60185', 'PDZIP_60186', 'PDZIP_60187', 'PDZIP_60188', 'PDZIP_60189',
             'PDZIP_60190', 'PDZIP_60191', 'PDZIP_60440', 'PDZIP_60502', 'PDZIP_60504',
             'PDZIP_60514', 'PDZIP_60515', 'PDZIP_60516', 'PDZIP_60517', 'PDZIP_60519',
             'PDZIP_60521', 'PDZIP_60527', 'PDZIP_60532', 'PDZIP_60540', 'PDZIP_60555',
             'PDZIP_60559', 'PDZIP_60561', 'PDZIP_60563', 'PDZIP_60564', 'PDZIP_60565',
             'PDZIP_60605', 'PDZIP_60607', 'PDZIP_60608', 'PDZIP_60609', 'PDZIP_60616',
             'PDZIP_60632', 'PDZIP_60635', 'PDZIP_60660',  'PDZIP_None']
    arr = [1 if zip.lower() in a else 0 for a in codes]
    if 1 not in np.unique(arr):
        arr[codes.index('PDZIP_None')] = 1
    return arr

def return_lang_encoded(language):
    arr = [0, 0, 0, 0, 0]
    if language is None:
        arr[4] = 1
    elif language.lower() == 'english':
        arr[0] = 1
    elif language.lower() == 'other':
        arr[1] = 1
    elif language.lower() == 'spanish':
        arr[2] = 1
    elif language.lower() == 'albanian':
        arr[3] = 1
    return arr

def select_features_visits(df):
    X = df[['PDAGE', 'PDHSIZE']]
    tools.convert_column_to_float(X, ['PDAGE', 'PDHSIZE'])
    X['PDHSIZE'] = X['PDHSIZE'].astype('float')
    langs = pd.get_dummies(df['PDLANG'], prefix='PDLANG')
    zipcode = pd.get_dummies(df['PDZIP'], prefix='PDZIP')
    categories = pd.get_dummies(df['PDBIRTH'], prefix='PDBIRTH')
    mstat = pd.get_dummies(df['PDMSTAT'], prefix='Mstat')
    edu = pd.get_dummies(df['PDEDU'], prefix='Edu')
    income = pd.get_dummies(df['PDINCOME'], prefix='income')
    emp = pd.get_dummies(df['PDEMP'], prefix='Emp')
    encodings = [langs, zipcode, categories, mstat, edu, income, emp]
    for e in encodings:
        for col in e.columns.values:
            X[col] = e[col]
    return X

def build_model(directory, filename):

    df = tools.read_file(directory, filename)
    labels = ['visit_ranges']
    print(select_features_visits(df).columns.values)
    X = select_features_visits(df).values
    Y = df[labels[0]].values
    n_inputs = X.shape[1]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.7, shuffle=True)

    cls = svm.SVC(kernel='linear',C=1).fit(x_train, y_train)

    return cls

def evaluate_visits_models(directory, filename):
    df = tools.read_file(directory, filename)
    labels = ['visit_ranges']

    X = select_features_visits(df).values

    Y = df[labels[0]].values
    n_inputs = X.shape[1]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2, shuffle=True)

    cls = svm.SVC(kernel='linear',C=1).fit(x_train, y_train)
    lin = svm.LinearSVC(C=1, max_iter=1000).fit(x_train, y_train)
    rbf = svm.SVC(kernel='rbf', gamma=0.5, C=1).fit(x_train, y_train)
    poly = svm.SVC(kernel='poly', degree=3, C=1).fit(x_train,y_train)

    cls_pred = cls.predict(X)
    lin_pred = lin.predict(X)
    poly_pred = poly.predict(X)
    rbf_pred = rbf.predict(X)

    cls_accuracy = accuracy_score(Y, cls_pred)
    cls_f1 = f1_score(Y, cls_pred, average='weighted')
    #print('Accuracy (Linear Kernel): ', "%.2f" % (cls_accuracy * 100))
    #print('F1 (Linear Kernel): ', "%.2f" % (cls_f1 * 100))

    lin_accuracy = accuracy_score(Y, lin_pred)
    lin_f1 = f1_score(Y, lin_pred, average='weighted')
    #print('Accuracy (Linear Kernel): ', "%.2f" % (lin_accuracy * 100))
    #print('F1 (Linear Kernel): ', "%.2f" % (lin_f1 * 100))

    poly_accuracy = accuracy_score(Y, poly_pred)
    poly_f1 = f1_score(Y, poly_pred, average='weighted')
    #print('Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy * 100))
    #print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1 * 100))

    rbf_accuracy = accuracy_score(Y, rbf_pred)
    rbf_f1 = f1_score(Y, rbf_pred, average='weighted')
    #print('Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy * 100))
    #print('F1 (RBF Kernel): ', "%.2f" % (rbf_f1 * 100))

    ## accuracy
    #print("acuracy:", metrics.accuracy_score(Y, y_pred=rbf_pred))
    # precision score
    #print("precision:", metrics.precision_score(Y, y_pred=rbf_pred,average=None,labels=['L','M','H','VH']))
    # recall score
    #print("recall", metrics.recall_score(Y, y_pred=rbf_pred,average=None,labels=['L','M','H','VH']))
    metric_dict = metrics.classification_report(Y, y_pred=rbf_pred,labels=['L','M','H','VH'], output_dict=True)
    pth = os.environ['PYTHONPATH'] + os.path.sep + 'files' + os.path.sep + 'rbfmetrics.csv'
    rbfmetrics = pd.DataFrame(metric_dict)
    rbfmetrics['metrics'] = ['Precision','Recall','F1 Score','Support']
    rbfmetrics.to_csv(path_or_buf=pth)


    # creating a confusion matrix
    #print(confusion_matrix(Y, cls_pred, labels=['L','M','H','VH']))
    linearsvc_con =confusion_matrix(Y, lin_pred, labels=['L','M','H','VH'])
    poly_con = confusion_matrix(Y, poly_pred, labels=['L', 'M', 'H', 'VH'])
    rbf_con = confusion_matrix(Y, rbf_pred, labels=['L', 'M', 'H', 'VH'])

    linearsvc_df =  pd.DataFrame(linearsvc_con, columns=['linsvc_L','linsvc_M','linsvc_H','linsvc_VH'])
    linearsvc_df['ranges'] = ['L','M','H','VH']
    poly_df = pd.DataFrame(poly_con, columns=['poly_L','poly_M','poly_H','poly_VH'])
    poly_df['ranges'] = ['L', 'M', 'H', 'VH']
    rbf_df = pd.DataFrame(rbf_con, columns=['rbf_L','rbf_M','rbf_H','rbf_VH'])
    rbf_df['ranges'] = ['L', 'M', 'H', 'VH']
    joined_df = pd.merge(rbf_df, poly_df, on='ranges')
    joined_df = pd.merge(joined_df, linearsvc_df, on='ranges')
    joined_df.set_index('ranges', inplace=True)
    joined_df['classes'] = ['L','M','H','VH']
    print(joined_df)
    pth = os.environ['PYTHONPATH'] + os.path.sep + 'files' + os.path.sep + 'visitmetrics.csv'
    joined_df.to_csv(path_or_buf=pth)

def test():
    evaluate_visits_models('files', 'dupage_chinatown.csv')

def predict(x):
    model = build_model('files', 'dupage_chinatown.csv')
    x = np.array(x)
    x = np.expand_dims(x, 0)
    pred = model.predict(x)
    return pred

if __name__ == '__main__':
    #x = [24, 3, 'spanish', '60613', 'us', 'single', 'graduate', 40000, 'full time employment']
    '''
    x = encode_input(24, 4, 'english', '60613', 'us', 'divorced/separated', 'graduate/professional degree', 40000, 'working')
    pred = predict(x)
    print(pred)
    '''
    test()