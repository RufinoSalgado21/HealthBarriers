import csv
from collections import Counter
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


def create_empty_column(df,name,length):
    df[name] = [' ']*length

def join_by_column(df,center,list_of_cols):
    lst = list_of_cols[:]
    new_df = df.groupby(center)[lst].agg(','.join).reset_index()
    return new_df

def write_to_txt_file(df, filename):
    path = os.environ['PYTHONPATH'] + os.path.sep + 'files' + os.path.sep + filename
    file1 = open(path, "w")
    file1.write(str(df.columns.values) + '\n')
    file1.write(str(df))
    file1.close()


'''
    Takes the given prediction numpy arrays and convert
    each predicted instance to zeroes except for the top three
    highest scoring, labeled as ones.
'''

def convert_to_binary(yhat, ytest):
   for index,row in enumerate(yhat):
       ones = np.count_nonzero(ytest[index])
       ind = np.argpartition(row, -ones)[-ones:]

       for i in range(len(row)):
           if i in ind:
               row[i] = 1
           else:
               row[i] = 0

'''
    Return a list with the top three labels based on the returned list from
    convert_to_prediction() which labels the top three predictions as 1, else 
    as 0.
'''
def return_predicted_labels(y, labels):
    output = []
    for index, row in enumerate(y):
        indices = [i for i in range(len(row)) if row[i] == 1] #Return list of labels predicted/true
        predictions = [ labels[j] for j in indices ]
        output.append(predictions)
    return output

def convert_column_to_float(X,column_list):
    for col in column_list:
        X[col] = X[col].replace('None', 0)
        X[col] = X[col].astype('float')

def my_filter(shape, dtype=None):
    kernel = np.zeros(shape)
    lst = [0]*shape[0]
    kernel[:,0, 0] = np.array([[1,0,1]])
    return kernel

'''
    Used by prediction methods to encode the most common labels
    represented by separate columns in a dataframe.
    Returns a new dataframe of of encoded labels with unique label
    names across the given column names listed.
    e.g. 
    input
    df[label1] : a, df[label2]: b, df[label3]: c
    output
    df[a, b, c] : [[1, 0, 0],[0, 1, 0],[0,0,1]]
'''
def multilabel_binary_label_encoding(df, labels):
    # Joined top three barriers into single column
    df['total_barriers'] = df[labels].apply(','.join, axis=1)

    # Split the three values in each row into a list for binarization
    for index, i in enumerate(df['total_barriers'].values):
        df['total_barriers'][index] = i.split(',')

    # MultiLabelBinarizer allows the binarization of multilabel values
    # Necessary to include all barriers listed by at least one patient without duplicating barriers
    mlb = MultiLabelBinarizer()
    pand = mlb.fit_transform(df['total_barriers'].values)

    temp_frame = pd.DataFrame(data=pand, columns=mlb.classes_)
    return mlb, temp_frame

def select_multilabels(df, labels):
    mlb, temp_frame = multilabel_binary_label_encoding(df, labels)
    Y = temp_frame.values
    return Y, mlb

'''
    Writes to output file the accuracy, precision, recall, and F1 Score for 
    the given confusion matrix as a string.
'''
def calculate_confusion_matrix_measures(confusion_matrices):
    trueN = confusion_matrices[0][0]
    trueP = confusion_matrices[1][1]
    falseP = confusion_matrices[0][1]
    falseN = confusion_matrices[1][0]
    acc = (trueP + trueN) / (trueP + falseP + trueN + falseN)
    if trueP == 0 and falseP == 0:
        prec = 0
    else:
        prec = trueP / (trueP + falseP)
    if trueP == 0 and falseN == 0:
        rec = 0
    else:
        rec = trueP / (trueP + falseN)
    if prec == 0 and rec == 0:
        f = 0
    else:
        f = 2 * (prec * rec) / (prec + rec)
    output = [acc,prec,rec,f]
    return output

#Replace NULL and NaN values with None
def replace_null_values(df):
    for col in df.columns.values:
        # df[col] = df[col].replace(['#NULL!'], None)
        df[col] = df[col].replace(['-'], '')
        df[col] = df[col].replace(['#NULL!'], np.nan)
        df[col] = df[col].replace(np.NaN, 'None')
        df[col] = df[col].replace(np.nan, 'None')
        df[col] = df[col].apply(str)
    print('Empty values replaced.')

# Utility method for removing several characters from a given string.
def replace_multi(string, list=[], replacement=''):
    for ls in list:
        string = string.replace(ls, replacement)
    return string


# Prints out the given dataframe to a csv file.
def write_to_csv(cols_list, csv_lists,directory, output_filename):
    path = os.environ['PYTHONPATH'] + os.path.sep + directory + os.path.sep + output_filename
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(cols_list)
        for c in csv_lists:
            try:
                writer.writerow(c)
            except:
                print(c)

def encode_PDBIRTH(df):
    for index, r in enumerate(df['PDBIRTH']):
        r = r.replace('Greece', 'GR')
        r = r.replace('Russia', 'RU')
        r = r.replace('Cuba', 'CU')
        r = r.replace('Bangladesh', 'BD')
        r = r.replace('mexico', 'MX')
        r = r.replace('Jamaica', 'JM')
        r = r.replace('El Salvador', 'SV')
        r = r.replace('Colombia', 'CO')
        r = r.replace('Chile', 'CL')
        r = r.replace('Bolivia', 'BO')
        r = r.replace('Mexico', 'MX')
        r = r.replace('Ecuador', 'EC')
        r = r.replace('Peru', 'PE')
        r = r.replace('Lebanon', 'LB')
        r = r.replace('United States', 'US')
        r = r.replace('Columbia', 'CO')
        r = r.replace('Argentina', 'AR')
        r = r.replace('Guatemala', 'GT')
        r = r.replace('Puerto Rico', 'PR')
        r = r.replace('Lithuania', 'LT')
        r = r.replace('Italy', 'IT')
        r = r.replace('Angola', 'AO')
        r = r.replace('Honduras', 'HN')
        r = r.replace('Iran', 'IR')
        r = r.replace('Poland', 'PL')
        r = r.replace('poland', 'PL')
        r = r.replace('Korea', 'None')
        r = r.replace('Philipenes', 'PH')
        r = r.replace('Phillipenes', 'PH')
        r = r.replace('Sudan', 'SD')
        r = r.replace('Jordan', 'JO')
        r = r.replace('Albania', 'AL')
        r = r.replace('India', 'IN')
        r = r.replace('Germany', 'DE')
        r = r.replace('Afhhanistan', 'AF')
        r = r.replace('???', 'None')
        r = r.replace('China', 'CN')
        r = r.replace('Vietnam', 'VN')
        df['PDBIRTH'][index] = r

def read_file(directory, filename):
    path = os.environ['PYTHONPATH'] + os.path.sep + directory + os.path.sep + filename
    pd.set_option('display.max_columns', None)
    dataset = pd.read_csv(path)
    print(filename + ' Dataset Read.')
    return dataset

def count_visits(df):
    df['visit_counts'] = [0]*len(df['R24Barrier'])
    for index, id in enumerate(df['R24Barrier'].values):
        count = Counter(id.split(','))
        visits = sum(count.values())
        df['visit_counts'][index] = visits

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()