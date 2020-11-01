from python import tools
import pandas as pd

def preprocess_chinatown(demographics_file,log_file):
    chinatown_demographics = tools.read_file('datasets',demographics_file)
    chinatown_logs = tools.read_file('datasets',log_file)

    chinatown_logs.rename(columns={'Preferred language': 'PDLANG'}, inplace=True)
    chinatown_logs.rename(columns={"Record ID (automatically assigned)": 'IDSUBJ'}, inplace=True)
    chinatown_demographics.rename(columns={'age  ': 'PDAGE'}, inplace=True)
    chinatown_demographics.rename(columns={'zip_code': 'PDZIP'}, inplace=True)
    chinatown_demographics.rename(columns={'record_id': 'IDSUBJ'}, inplace=True)
    chinatown_demographics.rename(columns={'native_land': 'PDBIRTH'}, inplace=True)
    chinatown_demographics.rename(columns={'marital status ': 'PDMSTAT'}, inplace=True)
    chinatown_demographics.rename(columns={'family_members_in_household':'PDHSIZE'}, inplace=True)
    chinatown_demographics.rename(columns={'education': 'PDEDU'}, inplace=True)
    chinatown_demographics.rename(columns={'household_income': 'PDINCOME'}, inplace=True)
    chinatown_demographics.rename(columns={'occupational status': 'PDEMP'}, inplace=True)

    chinatown_demographics = chinatown_demographics.sort_values(by='IDSUBJ')
    chinatown_logs['IDSUBJ'] = chinatown_logs['IDSUBJ'].astype(float)

    merged_dataset = pd.merge(chinatown_demographics, chinatown_logs, on='IDSUBJ')
    tools.replace_null_values(merged_dataset)
    #encode_PDLANG(chinatown_logs)
    remove_nonenglish_characters(merged_dataset)
    encode_features(merged_dataset)

    barriers =[]
    for col in merged_dataset.columns:
        if 'Barrier (choose one)' in col:
            barriers.append(col)
    merged_dataset['R24Barrier'] = merged_dataset[barriers[:]].agg(','.join, axis=1)

    actions = []
    for col in merged_dataset.columns:
        if 'Action taken (choose one)' in col:
            actions.append(col)
    merged_dataset['R24Action'] = merged_dataset[actions[:]].agg(','.join, axis=1)

    replace_barrier_names(merged_dataset)
    replace_action_names(merged_dataset)
    cols_list = merged_dataset.columns.values
    csv_lists = merged_dataset.values
    tools.write_to_csv(cols_list,csv_lists,'files','preprocessed_chinatown.csv')
    return merged_dataset

def replace_barrier_names(df):
    for index, r in enumerate(df['R24Barrier']):
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
        df['R24Barrier'][index] = r

def replace_action_names(df):
    for index, r in enumerate(df['R24Action']):
        r = r.replace('Accompaniment', 'AC1')
        r = r.replace('Scheduling appointment', 'SA1')
        r = r.replace('Referrals/ Direct Contact', 'RD1')
        r = r.replace('Education', 'E1')
        r = r.replace('Support', 'S1')
        r = r.replace('Arrangements', 'AR1')
        r = r.replace('Records/Recordkeeping', 'R1')
        r = r.replace('Action pending / No action', 'Pending')
        df['R24Action'][index] = r

def encode_features(df):
    for index, col in enumerate(df.columns):
        if col == 'PDLANG':
            encode_PDLANG(df)
        elif col == 'PDMSTAT':
            df[col] = df[col].replace(['Single Never Married'], 1)
            df[col] = df[col].replace(['Married'], 2)
            df[col] = df[col].replace(['Cohabit'], 2)
            df[col] = df[col].replace(['Divorced'], 3)
            df[col] = df[col].replace(['Separated'], 3)
            df[col] = df[col].replace(['Widowed'], 4)
            df[col] = df[col].replace(['Refuse to Answer\<em\>'], 98)
            df[col] = df[col].replace(['Refuse to Answer</em>'], 98)
        elif col == 'PDEDU':
            df[col] = df[col].replace(['Primary School'], 1)
            df[col] = df[col].replace(['Junior High'], 1)
            df[col] = df[col].replace(['Some high school'], 2)
            df[col] = df[col].replace(['High School'], 3)
            df[col] = df[col].replace(['Vocational or technical school after high school'], 4)
            df[col] = df[col].replace(['Attended college but did not graduate'], 4)
            df[col] = df[col].replace(['Associate degree'], 5)
            df[col] = df[col].replace(['University Graduate'], 6)
            df[col] = df[col].replace(['Graduate or professional degree'], 7)
            df[col] = df[col].replace(['I have no education'], 98)
        elif col == 'PDBIRTH':
            tools.encode_PDBIRTH(df)
        elif col == 'PDINCOME':
            df[col] = df[col].replace(["$0 - $9,999"], 1)
            df[col] = df[col].replace(["$10,000 - $14,999"], 2)
            df[col] = df[col].replace(["$15,000 - $19,999"], 2)
            df[col] = df[col].replace(["$20,000 - $34,999"], 98)
            df[col] = df[col].replace(["$35,000 - $49,999"], 98)
            df[col] = df[col].replace(['Do Not Know'], 98)
            df[col] = df[col].replace(["$50,000 - $74,999"], 6)
            df[col] = df[col].replace(["$75,000 - $99,999"], 6)
        elif col == 'PDEMP':
            df[col] = df[col].replace(['Working'],98)
            df[col] = df[col].replace(['Retired'],0)
            df[col] = df[col].replace(['Housewife'],0)
            df[col] = df[col].replace(['Unemployed'], 0)

def remove_nonenglish_characters(df):
    for index, r in enumerate(df['PDINCOME'].values):
        if '\u81f3' in r:
            s = r.split('\u81f3')
            s = str(s[0]) + '-' + str(s[1])
            s = s.lstrip('<em>')
            df['PDINCOME'][index] = s
    for index, r in enumerate(df['born_in_US']):
        if 'Yes' in r:
            df['PDBIRTH'][index] = 'United States'
    for index, r in enumerate(df['PDBIRTH']):
        if r != 'United States' and r != 'Vietnam' and r != '???':
            df['PDBIRTH'][index] = 'China'
    for index, r in enumerate(df['born_in_US']):
        if len(r) > 3:
            r = 'Yes'
        df['born_in_US'][index] = r


def encode_PDLANG(df):
    lang_codes = ['English', 'Spanish', 'Haitian Creole', 'Vietnamese',
                  'Portuguese Creole', 'Albanian', 'Cambodian', 'Russian',
                  'Somali', 'Other']
    col = df['PDLANG']
    df['PDLANG'] = df['PDLANG'].replace(['Mandarin'], 'Other')
    df['PDLANG'] = df['PDLANG'].replace(['Cantonese'], 'Other')
    df['PDLANG'] = df['PDLANG'].replace(['Toishanese'], 'Other')
    for l in lang_codes:
        df['PDLANG'] = df['PDLANG'].replace([l], lang_codes.index(l) + 1)


#if __name__ == '__main__':
    #preprocess_chinatown('df_all_INFO.csv','Tracking Log [All] #1-10 for NEIU (2).csv')