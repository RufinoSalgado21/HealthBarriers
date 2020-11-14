from collections import Counter

from python import tools
import keras
import numpy as np
import pandas as pd

def preprocess_dupage(demographics_file,log_file):
    #Read datasets into dataframes
    dupage_demographics = tools.read_file('datasets',demographics_file)
    dupage_logs = tools.read_file('datasets',log_file)

    #Remove null values
    tools.replace_null_values(dupage_logs)
    tools.replace_null_values(dupage_demographics)

    #Group rows in dupage_log by subject ID
    dupage_logs = tools.join_by_column(dupage_logs,'IDSUBJ',['R24Barrier','R24Action'])

    #Fill visit_counts column with total number of barriers logged per subject.
    tools.create_empty_column(dupage_logs, 'visit_counts', len(dupage_logs['IDSUBJ']))
    count_visits(dupage_logs)

    inner_merged = pd.merge(dupage_logs,dupage_demographics, on=['IDSUBJ'])
    tools.encode_PDBIRTH(inner_merged)

    cols_list = inner_merged.columns.values
    csv_lists = inner_merged.values
    tools.write_to_csv(cols_list,csv_lists,'files','preprocessed_dupage.csv')
    return inner_merged

def count_visits(dupage_logs):
    for index, id in enumerate(dupage_logs['R24Barrier'].values):
        count = Counter(id.split(','))
        visits = sum(count.values())
        dupage_logs['visit_counts'][index] = visits

#if __name__ == '__main__':
    #preprocess_dupage('DuPage_demographics_v2.csv','DuPage_tracking_log.csv')