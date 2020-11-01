from collections import Counter

from python import tools,dupage_preprocessing,chinatown_preprocessing
import pandas as pd

def preprocess(df_dupage,df_chinatown):

    frames = [df_dupage, df_chinatown]
    merged_dataset = pd.concat(frames, join='inner', ignore_index=True)

    tools.count_visits(merged_dataset)

    for index, col in enumerate(merged_dataset['R24Barrier'].values):
        count = Counter(id.split(','))
        

    cols_list = merged_dataset.columns.values
    csv_list = merged_dataset.values
    tools.write_to_csv(cols_list,csv_list,'files','dupage_chinatown.csv')


if __name__ == '__main__':
    df_dupage = dupage_preprocessing.preprocess_dupage('DuPage_demographics_v2.csv','DuPage_tracking_log.csv')
    df_chinatown = chinatown_preprocessing.preprocess_chinatown('df_all_INFO.csv','Tracking Log [All] #1-10 for NEIU (2).csv')
    preprocess(df_dupage, df_chinatown)