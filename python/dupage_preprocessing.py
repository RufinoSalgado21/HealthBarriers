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
    encode_income_levels(inner_merged)
    encode_emp_status(inner_merged)

    replace_barrier_names(inner_merged)
    replace_action_names(inner_merged)

    cols_list = inner_merged.columns.values
    csv_lists = inner_merged.values
    tools.write_to_csv(cols_list,csv_lists,'files','preprocessed_dupage.csv')
    return inner_merged

def encode_income_levels(df):
    for index, i in enumerate(df['PDINCOME']):
        df.at[index, 'PDINCOME'] = round(float(i))

def encode_emp_status(df):
    #Housewife, Working, Retired, Unemployed, Other, Student
    for index, i in enumerate(df['PDEMP']):
        if i is None:
            df.at[index, 'PDEMP'] = 'Other'
        elif i == '0':
            df.at[index, 'PDEMP'] = 'Unemployed'
        elif i == '1' or i == '2':
            df.at[index, 'PDEMP'] = 'Working'
        else:
            df.at[index, 'PDEMP'] = 'Other'

def count_visits(dupage_logs):
    for index, id in enumerate(dupage_logs['R24Barrier'].values):
        count = Counter(id.split(','))
        visits = sum(count.values())
        dupage_logs['visit_counts'][index] = visits

def replace_action_names(df):
    for index, r in enumerate(df['R24Action']):
        #r = r.replace('1', 'Other')
        #r = r.replace('4', 'Other')
        r = r.replace('No Action', 'None')
        r = r.replace('Pending', 'Other')
        r = r.replace('Action Other', 'Other')
        r = r.replace('Action Pending', 'Other')
        r = r.replace('Travel', 'Other')
        df['R24Action'][index] = r

def replace_barrier_names(df):
    for index, r in enumerate(df['R24Barrier']):
        r = r.replace('AP3', 'Other')
        r = r.replace('CC4', 'Other')
        r = r.replace('CM1', 'CM')
        r = r.replace('FP5', 'Other')
        r = r.replace('FP6', 'Other')
        r = r.replace('I8', 'Other')
        r = r.replace('l1', 'L1')
        r = r.replace('l7', 'Other')
        r = r.replace('Ll1', 'LI1')
        r = r.replace('LL1', 'LI1')
        r = r.replace('Ll2', 'LI2')
        r = r.replace('L12', 'LI2')
        r = r.replace('MM9', 'Other')
        r = r.replace('01', 'O1')
        r = r.replace('PB6', 'Other')
        r = r.replace('PN6', 'Other')
        r = r.replace('RD2', 'Other')
        r = r.replace('SA1', 'Other')
        r = r.replace('SS9', 'Other')
        r = r.replace('RD2', 'Other')
        r = r.replace('W3', 'Other')
        if 'T1 & T2' in r:
            r = r.replace('T1 & T2','T1')
            r += 'T2'
        df['R24Barrier'][index] = r

if __name__ == '__main__':
    preprocess_dupage('DuPage_demographics_v2.csv','DuPage_tracking_log.csv')