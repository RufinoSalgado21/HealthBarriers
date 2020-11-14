from collections import Counter

from python import tools,dupage_preprocessing,chinatown_preprocessing
import pandas as pd

def preprocess(df_dupage,df_chinatown):

    frames = [df_dupage, df_chinatown]
    merged_dataset = pd.concat(frames, join='inner', ignore_index=True)

    tools.count_visits(merged_dataset)

    barriers = ['barrier_1','barrier_2','barrier_3']
    col_len = len(merged_dataset['IDSUBJ'])
    print(col_len)
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

    cols_list = merged_dataset.columns.values
    csv_list = merged_dataset.values
    tools.write_to_csv(cols_list,csv_list,'files','dupage_chinatown.csv')


if __name__ == '__main__':
    df_dupage = dupage_preprocessing.preprocess_dupage('DuPage_demographics_v2.csv','DuPage_tracking_log.csv')
    df_chinatown = chinatown_preprocessing.preprocess_chinatown('df_all_INFO.csv','Tracking Log [All] #1-10 for NEIU (2).csv')
    preprocess(df_dupage, df_chinatown)