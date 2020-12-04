from collections import Counter
from python import tools,dupage_preprocessing,chinatown_preprocessing
import pandas as pd
import numpy as np

def create_feature_visit_ranges(df, visit_count_column_name, new_column_name):
    df[new_column_name] = ['']*df.values.shape[0]
    mean = np.mean(df[visit_count_column_name].values)
    stdv = np.std(df[visit_count_column_name].values)
    low_threshold = mean-stdv
    mid_threshold = mean
    high_threshold = mean+stdv
    for index, row in enumerate(df[visit_count_column_name]):
        if row <= low_threshold:
            df.at[index, new_column_name] = 'L'
        elif row <= mid_threshold:
            df.at[index, new_column_name] = 'M'
        elif row <= high_threshold:
            df.at[index, new_column_name] = 'H'
        else:
            df.at[index, new_column_name] = 'VH'


def preprocess(df_dupage,df_chinatown):
    frames = [df_dupage, df_chinatown]
    merged_dataset = pd.concat(frames, join='inner', ignore_index=True)

    tools.count_visits(merged_dataset)
    create_feature_visit_ranges(merged_dataset, 'visit_counts','visit_ranges')
    barriers = ['barrier_1','barrier_2','barrier_3']
    col_len = len(merged_dataset['IDSUBJ'])

    tools.create_empty_column(merged_dataset, barriers[0], col_len)
    tools.create_empty_column(merged_dataset, barriers[1], col_len)
    tools.create_empty_column(merged_dataset, barriers[2], col_len)

    actions = ['action_1','action_2','action_3']
    tools.create_empty_column(merged_dataset, actions[0], col_len)
    tools.create_empty_column(merged_dataset, actions[1], col_len)
    tools.create_empty_column(merged_dataset, actions[2], col_len)

    for index, row in enumerate(merged_dataset['R24Barrier'].values):
        count = Counter(row.split(','))
        c = {k: v for k, v in sorted(count.items(), key=lambda item: item[1])}
        lst = list(c.keys())
        for i, bar in enumerate(barriers):
            l = len(lst)-1-i
            if l < 0:
                l = 0
            merged_dataset[bar][index] = lst[l]

    for index, row in enumerate(merged_dataset['R24Action'].values):
        count = Counter(row.split(','))
        c = {k: v for k, v in sorted(count.items(), key=lambda item: item[1])}
        lst = list(c.keys())
        for i, act in enumerate(actions):
            l = len(lst)-1-i
            if l < 0:
                l = 0
            merged_dataset[act][index] = lst[l]

    for action in actions:
        for index, row in enumerate(merged_dataset[action]):
            if row == '1':
                merged_dataset.at[index, action] = 'Other'
            if row == '4':
                merged_dataset.at[index, action] = 'Other'

    hsize_mean = calc_mean_sans_none(merged_dataset, 'PDHSIZE')
    age_mean = calc_mean_sans_none(merged_dataset, 'PDAGE')

    for index, row in enumerate(merged_dataset['PDHSIZE']):
        if row == 'None':
            merged_dataset.at[index, 'PDHSIZE'] = hsize_mean
    for index, row in enumerate(merged_dataset['PDAGE']):
        if row == 'None':
            merged_dataset.at[index, 'PDAGE'] = age_mean
    for index, row in enumerate(merged_dataset['PDEDU']):
        if row == 'None':
            merged_dataset.at[index, 'PDEDU'] = '98'
    for index, row in enumerate(merged_dataset['PDMSTAT']):
        if row == 'None':
            merged_dataset.at[index, 'PDMSTAT'] = '98'

    cols_list = merged_dataset.columns.values
    csv_list = merged_dataset.values
    tools.write_to_csv(cols_list,csv_list,'files','dupage_chinatown.csv')

    return merged_dataset


def calc_mean_sans_none(merged_dataset, column):
    sum = 0
    count = 0
    for val in merged_dataset[column].values:
        if val != 'None':
            sum += float(val)
            count += 1
    mean = sum / count
    return mean


if __name__ == '__main__':
    print('PREPROCESSING DUPAGE...')
    df_dupage = dupage_preprocessing.preprocess_dupage('DuPage_demographics_v2.csv', 'DuPage_tracking_log.csv')
    print('PREPROCESSING CHINATOWN...')
    df_chinatown = chinatown_preprocessing.preprocess_chinatown('df_all_INFO.csv',
                                                                'Tracking Log [All] #1-10 for NEIU (2).csv')

    print('MERGING DATASETS INTO ONE...')
    merged = preprocess(df_dupage, df_chinatown)