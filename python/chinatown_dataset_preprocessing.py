import csv
import difflib
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

def clean_up(df):
    for col in df.columns.values:
        df[col] = df[col].replace(['Checked'], 1)
        df[col] = df[col].replace(['Unchecked'], 0)
        df[col] = df[col].replace(np.NaN, 'None')
        if 'Preferred language' in col:
            df[col] = df[col].replace(['English'], 1)
            df[col] = df[col].replace(['Cantonese'], 11)
            df[col] = df[col].replace(['Mandarin'], 12)
            df[col] = df[col].replace(['Toishanese'], 13)

def count_uniques(col, new_df, num_of_most_common=0):
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


def append_ranked_values(col, index, length, new_df, num_of_most_common, sorted_vals):
    if num_of_most_common > 0:
        for n in range(num_of_most_common):
            new_col = col + '_' + str(n)
            if len(sorted_vals) > n:
                new_df[new_col][index] = sorted_vals[n][0]
            else:
                new_df[new_col][index] = sorted_vals[len(sorted_vals) - 1][0]


def remove_extra_characters(dataset_second):
    for index, r in enumerate(dataset_second['household_income'].values):
        if '\u81f3' in r:
            s = r.split('\u81f3')
            s = str(s[0]) + '-' + str(s[1])
            if '<em>' in s:
                s = s.lstrip('<em>')
            dataset_second['household_income'][index] = s

def replace_multi(string, list=[], replacement=''):
    for ls in list:
        string = string.replace(ls,replacement)
    return string

def write_to_csv(merged_dataset):
    csv_lists = merged_dataset.values.tolist()
    cols_list = merged_dataset.columns.values.tolist()

    path = os.environ['PYTHONPATH'] + os.path.sep + 'files' + os.path.sep + 'merged_chinatown.csv'
    with open(path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(cols_list)
        for c in csv_lists:
            c[253] = replace_multi(str(c[253]), ['[', ']', '\'', ','], ',')
            c[254] = replace_multi(str(c[254]), ['[', ']', '\'', ','], ',')
            writer.writerow(c)

def merge_actions(merged_dataset):
    merged_dataset['R24Barrier'] = merged_dataset[
        ['Barrier (choose one)', 'Barrier (choose one).1', 'Barrier (choose one).2', 'Barrier (choose one).3',
         'Barrier (choose one).4', 'Barrier (choose one).5', 'Barrier (choose one).6', 'Barrier (choose one).7',
         'Barrier (choose one).8', 'Barrier (choose one).9'
         ]].agg(','.join, axis=1)


def merge_barriers(merged_dataset):
    merged_dataset['R24Action'] = merged_dataset[
        ['Action taken (choose one)', 'Action taken (choose one).1', 'Action taken (choose one).2',
         'Action taken (choose one).3',
         'Action taken (choose one).4', 'Action taken (choose one).5', 'Action taken (choose one).6',
         'Action taken (choose one).7',
         'Action taken (choose one).8', 'Action taken (choose one).9'
         ]].agg(','.join, axis=1)

def main():
    dataset_first = read_file('datasets', 'Tracking Log [All] #1-10 for NEIU (2).csv')
    dataset_second = read_file('datasets','dF_all_INFO.csv')

    clean_up(dataset_first)
    clean_up(dataset_second)
    dataset_first['Record ID (automatically assigned)'] = dataset_first['Record ID (automatically assigned)'].astype(float)
    dataset_first.rename(columns={'Record ID (automatically assigned)':'record_id'}, inplace=True)

    #Removing <emp> and different language character
    remove_extra_characters(dataset_second)

    #Merging both chinatown datasets based on record ids
    merged_dataset = pd.merge(dataset_first, dataset_second, on='record_id')
    merged_dataset.rename(columns={'Preferred language' : 'PDLANG'}, inplace=True)
    merged_dataset.rename(columns={'age  ': 'PDAGE'}, inplace=True)
    merged_dataset.rename(columns={'marital status ': 'PDSTAT'}, inplace=True)
    merged_dataset.rename(columns={'zip_code': 'PDZIP'}, inplace=True)
    merged_dataset.rename(columns={'record_id': 'IDSUBJ'}, inplace=True)
    merged_dataset.rename(columns={'native_land': 'PDBIRTH'}, inplace=True)

    merge_actions(merged_dataset)
    merge_barriers(merged_dataset)
    count_uniques('R24Barrier',merged_dataset)
    count_uniques('R24Action',merged_dataset)
    for index,r in enumerate(merged_dataset['born_in_US']):
        if 'Yes' in r:
            merged_dataset['PDBIRTH'][index] = 'United States'
    for index,r in enumerate(merged_dataset['PDBIRTH']):
        if r != 'United States' and r != 'Vietnam' and r != '???':
            merged_dataset['PDBIRTH'][index] = 'China'


    print(merged_dataset.columns.values)

    "If 'Other' barrier, please write in:" "If 'Other' barrier, please write in:.1" "If 'Other' barrier, please write in:.2"
    "If 'Other' barrier, please write in:.3" "If 'Other' barrier, please write in:.4" "If 'Other' barrier, please write in:.5"
    "If 'Other' barrier, please write in:.6" "If 'Other' barrier, please write in:.7"
    "If 'Other' barrier, please write in:.8" "If 'Other' barrier, please write in:.9"

    #Writing the merged list to a csv file
    write_to_csv(merged_dataset)
    lst = merged_dataset['PDBIRTH'].tolist()
    provinces = ['Guangdong', 'Shandong', 'Henan', 'Sichuan', 'Jiangsu', 'Hebei', 'Hunan', 'Anhui', 'Hubei', 'Zhejiang',
                 'Guangxi', 'Yunnan', 'Jiangxi',
                 'Liaoning', 'Fujian', 'Shaanxi', 'Heilongjiang', 'Shanxi', 'Guizhou', 'Chongqing', 'Jilin', 'Gansu',
                 'Inner Mongolia',
                 'Xinjiang', 'Shanghai', 'Beijing', 'Tianjin', 'Hainan', 'Hong Kong', 'Ningxia', 'Qinghai', 'Tibet',
                 'Macau']

    print(merged_dataset['born_in_US'])
    print(set(lst))



if __name__ == '__main__':
    main()