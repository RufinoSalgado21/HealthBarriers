import os, csv

import keras
import pandas as pd
import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


def read_file(directory,filename):
    path = os.environ['PYTHONPATH'] + os.path.sep + directory + os.path.sep + filename
    #pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    dataset = pd.read_csv(path)
    print(filename + ' Dataset Read.')
    return dataset

def replace_NaN(X):
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(X)
    X = imputer.transform(X)
    return X

def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(18, input_dim=18, activation='relu'))
	model.add(Dense(3, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def write_to_txt_file(df, filename):
    path = os.environ['PYTHONPATH'] + os.path.sep + 'files' + os.path.sep + filename
    file1 = open(path, "w")
    file1.write(str(df.columns.values) + '\n')
    file1.write(str(df))
    file1.close()

#Updates the given column of a dataframe to a list of unique values at each row
def uniques(col, df):
    for index, i in enumerate(df[col]):
        df[col][index] = list(set(i.split(' ')))

#Replace NULL and NaN values with None
def replace_null_values(df):
    for col in df.columns.values:
        # df[col] = df[col].replace(['#NULL!'], None)
        df[col] = df[col].replace(['-'], '')
        df[col] = df[col].replace(['#NULL!'], np.nan)
        df[col] = df[col].replace(np.nan, 'None')
        df[col] = df[col].apply(str)
    print('Empty values replaced.')

def replace_multi(string, list=[], replacement=''):
    for ls in list:
        string = string.replace(ls,replacement)
    return string

def count_uniques(col, new_df, num_of_most_common):
    length = len(new_df[col])
    for n in range(num_of_most_common):
        new_col = col + '_' + str(n)
        new_df[new_col] = ['x'] * length

    for index, r in enumerate(new_df[col]):
        unqs = list(set(r.split(',')))
        zeroes = [0] * len(unqs)
        dct = dict(zip(unqs, zeroes))

        for i in dct.keys():
            c = r.count(i)
            dct[i] = c

        sorted_vals = sorted(dct.items(), key=lambda x: x[1], reverse=True)
        append_ranked_values(col, index, length, new_df, num_of_most_common, sorted_vals)
    '''
        lst = []
        for k in dct.keys():
            lst.append(str(k) + '-' + str(dct[k]))
        new_df[col][index] = lst
    '''


def append_ranked_values(col, index, length, new_df, num_of_most_common, sorted_vals):
    if num_of_most_common > 0:
        for n in range(num_of_most_common):
            new_col = col + '_' + str(n)
            if len(sorted_vals) > n:
                new_df[new_col][index] = sorted_vals[n][0]
            else:
                new_df[new_col][index] = sorted_vals[len(sorted_vals) - 1][0]


def main():
    file = 'DuPage_tracking_log.csv'
    file_demographics = "DuPage_patient_demographics.csv"
    df = read_file('datasets',file)
    df_demographics = read_file('datasets',file_demographics)

    replace_null_values(df)
    replace_null_values(df_demographics)

    # Group rows by subject id values
    #['LOGDATE','LOGTYPE','LOGENC','R24Barrier','R24Action','LOGACTT','LOGTYPEO','LOGTIME','LOGBARID','LOGBARO','LOGACTID','LOGACTO','LOGACTD']
    new_df = df.groupby('IDSUBJ')['R24Barrier','R24Action'].agg(','.join).reset_index()

    print('Rows grouped by subject IDs.')

    #Remove duplicates in each row of each column

    count_uniques('R24Barrier', new_df,2)
    count_uniques('R24Action', new_df, 2)
    print(new_df['R24Barrier'])
    print('Duplicate values in columns removed.')

    #Inner merge two datasets, returning df of items with all features available
    inner_merged = pd.merge(new_df, df_demographics, on=['IDSUBJ'])
    #write_to_txt_file(inner_merged, 'merged.txt')

    #Create 2 lists, containing the column name values and each row's values
    cols_list = inner_merged.columns.values.tolist()
    csv_lists = inner_merged.values.tolist()

    #Write new lists to csv file
    write_to_csv(cols_list, csv_lists)

    merged_df = read_file('files','merged.csv')
    merged_np = merged_df.values.tolist()
    #split r24 columns by space
    for row in merged_np:
        row[1] = row[1].split(' ')
        row[2] = row[2].split(' ')

    features = merged_df.iloc[:,7:25].values
    #print(merged_df.iloc[:,7:25])
    labels_barrier = merged_df.iloc[:,3]
    labels_action = merged_df.iloc[:,5]
    #print(features)
    #print(labels_barrier)
    #print(labels_action)

    encoder = LabelEncoder()
    encoder.fit(labels_barrier)
    encoded_Y = encoder.transform(labels_barrier)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_barrier = np_utils.to_categorical(encoded_Y)
    #merged_df['R24Barrier'] = le.fit_transform(merged_df['R24Barrier'])
    print(len(dummy_barrier[0]))
    print(dummy_barrier)

    le = LabelEncoder()
    cols = [0,2,3,7,8,16]
    #print(features[0])
    for c in cols:
        features[:,c] = le.fit_transform(features[:,c])
    #print(features[0])

    #ML modeling
    #18 features beyond most and second most common values
    #print(merged_np)


def write_to_csv(cols_list, csv_lists):
    path = os.environ['PYTHONPATH'] + os.path.sep + 'files' + os.path.sep + 'merged.csv'
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(cols_list)
        for c in csv_lists:
            c[1] = replace_multi(str(c[1]), ['[', ']', '\'', ','], ',')
            c[2] = replace_multi(str(c[2]), ['[', ']', '\'', ','], ',')
            writer.writerow(c)


if __name__ == '__main__':
    main()