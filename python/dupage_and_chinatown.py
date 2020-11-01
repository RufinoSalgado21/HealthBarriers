import csv
import os
import pandas as pd
from geopy.geocoders import GoogleV3
import geopy.distance
import googlemaps
import numpy as np
from python import web_scrapper
from python import distance_calculator

#Reads in a given csv file and passes it to a dataframe instance.
def read_file(directory,filename):
    path = os.environ['PYTHONPATH'] + os.path.sep + directory + os.path.sep + filename
    #pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    dataset = pd.read_csv(path)
    print(filename + ' Dataset Read.')
    return dataset

#Utility method for removing several characters from a given string.
def replace_multi(string, list=[], replacement=''):
    for ls in list:
        string = string.replace(ls,replacement)
    return string

#Prints out the given dataframe to a csv file.
def write_to_csv(cols_list, csv_lists, barriers_indx, actions_indx):
    path = os.environ['PYTHONPATH'] + os.path.sep + 'files' + os.path.sep + 'merged_dupage_chinatown.csv'
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(cols_list)
        for c in csv_lists:
            #c[1].sort()
            #c[2].sort()
            c[barriers_indx] = replace_multi(str(c[barriers_indx]), ['[', ']', '\'', ','], '')
            c[actions_indx] = replace_multi(str(c[actions_indx]), ['[', ']', '\'', ','], '')
            writer.writerow(c)
'''
    Appends the count of appearances of unique values in the given column of a dataframe.
    Third, parameter creates additional columns equal to the given value holding the most
    common values in decreasing order.
'''
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

        lst = []
        for k in dct.keys():
            lst.append(str(k) + '-' + str(dct[k]))
        new_df[col][index] = lst

'''
    Secondary method to count_uniques. Creates a number of additional, new columns
    corresponding to the given number of columns passed as a parameter. The columns 
    hold the most common values counted in decreasing order.
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
    df = read_file('files','merged.csv')
    df2 = read_file('files','merged_chinatown.csv')

    frames = [df, df2]
    joined_frames = pd.concat(frames, join='inner', ignore_index=True)
    replace_barrier_names(joined_frames)
    replace_action_names(joined_frames)
    clean_PDBIRTH(joined_frames)

    count_uniques('R24Barrier',joined_frames,3)
    count_uniques('R24Action', joined_frames, 3)


    check_for_duplicates(joined_frames,'PDBIRTH')
    #zipcode = search.main('60641')
    #print(zipcode)
    length = len(joined_frames['PDZIP'])
    joined_frames['Nearest_hospital'] = ['x']*length
    joined_frames['Km_to_nearest_hospital'] = ['x']*length

    #This section of code iterates throught the zipcodes, searching for the nearest hospitals and calculating their
    #distance in kilometers. The name of the nearest hospital and the distance to it are entered in two new corresponding columns.

    for index,row in enumerate(joined_frames['PDZIP']):
        if row is None or row == 'None':
            joined_frames['Nearest_hospital'][index] = 'None'
            joined_frames['Km_to_nearest_hospital'][index] = 0
            continue

        distance = 0

        try:
            hospital = distance_calculator.find_nearest_hospital(row)
            zipcode = distance_calculator.find_lat_lng(row)
            distance = distance_calculator.calculate_distance(zipcode, hospital)
            joined_frames['Nearest_hospital'][index] = hospital[0]
            joined_frames['Km_to_nearest_hospital'][index] = distance
        except:
            joined_frames['Nearest_hospital'][index] = 'None'
            joined_frames['Km_to_nearest_hospital'][index] = 0
            continue

    X = joined_frames[['PDAGE', 'Km_to_nearest_hospital']]
    for col in X.columns.values:
        X[col] = X[col].replace('None', '0')
    X['PDAGE'] = X['PDAGE'].astype('float')

    cols_list = joined_frames.columns.values.tolist()
    csv_list = joined_frames.values.tolist()
    write_to_csv(cols_list, csv_list, joined_frames.columns.get_loc('R24Barrier'),
                 joined_frames.columns.get_loc('R24Action'))


#Removes duplicated countries from column PDBIRTH due to grammatical errors.
def clean_PDBIRTH(joined_frames):
    for index, r in enumerate(joined_frames['PDBIRTH']):
        if '???' in r:
            joined_frames['PDBIRTH'][index] = 'None'
        elif 'Phil' in r:
            joined_frames['PDBIRTH'][index] = 'Philippines'
        elif 'exico' in r:
            joined_frames['PDBIRTH'][index] = 'Mexico'
        elif 'oland' in r:
            joined_frames['PDBIRTH'][index] = 'Poland'
        elif 'Afhh' in r:
            joined_frames['PDBIRTH'][index] = 'Afghanistan'

#Return a list of unique value that appear in a given column.
def check_for_duplicates(joined_frames,col_name):
    lst = list(set(joined_frames[col_name].values.tolist()))
    lst.sort()
    print(lst)

#Reassigns column values to match action codes.
def replace_action_names(joined_frames):
    for index, r in enumerate(joined_frames['R24Action']):
        r = r.replace('Accompaniment', 'AC1')
        r = r.replace('Scheduling appointment', 'SA1')
        r = r.replace('Referrals/ Direct Contact', 'RD1')
        r = r.replace('Education', 'E1')
        r = r.replace('Support', 'S1')
        r = r.replace('Arrangements', 'AR1')
        r = r.replace('Records/Recordkeeping', 'R1')
        r = r.replace('Action pending / No action', 'Pending')
        joined_frames['R24Action'][index] = r

#Reassigns columns values to match barrier codes
def replace_barrier_names(joined_frames):
    for index, r in enumerate(joined_frames['R24Barrier']):
        r = r.replace('Language/interpreter', 'LI1')
        r = r.replace('Fear', 'F1')
        r = r.replace('Citizenship','CC1')
        r = r.replace('Family/community issues', 'FC1')
        r = r.replace('Transportation', 'T1')
        r = r.replace('Work schedule conflicts', 'W1')
        r = r.replace('Out of town/country', 'O1')
        r = r.replace('Financial problems', 'FP1')
        r = r.replace('Navigator barriers', 'N1')
        r = r.replace("Other (write-in)", 'Other')
        r = r.replace('Communication concers with medical personnel', 'CP1')
        r = r.replace('Medical and mental health co-morbidity', 'MM1')
        r = r.replace('Perceptions/Beliefs about tests/treatment', 'PB1')
        r = r.replace('Insurance/Uninsured/Underinsured', 'I1')
        r = r.replace('Social/practical support', 'SP1')
        r = r.replace('System problems with scheduling care', 'SS1')
        joined_frames['R24Barrier'][index] = r


if __name__ == '__main__':
    main()