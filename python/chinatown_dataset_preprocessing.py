import csv
import os
import pandas as pd
import numpy as np

def read_file(directory,filename):
    path = os.environ['PYTHONPATH'] + os.path.sep + directory + os.path.sep + filename
    #pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    dataset = pd.read_csv(path)
    print(filename + ' Dataset Read.')
    return dataset

def main():
    dataset_first = read_file('datasets', 'Tracking Log [All] #1-10 for NEIU (2).csv')
    dataset_second = read_file('datasets','dF_all_INFO.csv')

    clean_up(dataset_first)
    clean_up(dataset_second)
    dataset_first['Record ID (automatically assigned)'] = dataset_first['Record ID (automatically assigned)'].astype(float)
    dataset_first.rename(columns={'Record ID (automatically assigned)':'record_id'}, inplace=True)

    for index,r in enumerate(dataset_second['household_income'].values):
        if '\u81f3' in r:
            s = r.split('\u81f3')
            #print(s[1])
            dataset_second['household_income'][index]= str(s[0]) + '-' + str(s[1])
        elif '<em>' in r:
            s = r.lstrip('<em')
            dataset_second['household_income'][index] = s
    print(dataset_second['household_income'])

    merged_dataset = pd.merge(dataset_first, dataset_second, on='record_id')

    csv_lists = merged_dataset.values.tolist()
    cols_list = merged_dataset.columns.values.tolist()
    path = os.environ['PYTHONPATH'] + os.path.sep + 'files' + os.path.sep + 'merged_chinatown.csv'
    with open(path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(cols_list)
        for c in csv_lists:
            writer.writerow(c)


def clean_up(df):
    for col in df.columns.values:
        df[col] = df[col].replace(['Checked'], 1)
        df[col] = df[col].replace(['Unchecked'], 0)
        df[col] = df[col].replace(np.NaN, 'None')

    #remove \u81f3 from income range col

if __name__ == '__main__':
    main()