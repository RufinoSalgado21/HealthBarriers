import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from python import multilabel_classification, tools

def plot_correlations(df):
    plt.figure(figsize=(25, 20))
    sns.set(font_scale=0.5)
    data = df.copy()
    labels = ['barrier_1','barrier_2','barrier_3']
    labels_actions = ['action_1','action_2','action_3']
    X = multilabel_classification.select_features(data)
    Y, mlb = tools.select_multilabels(data,labels)
    columns = X.columns.values

    newX = pd.DataFrame(X, columns=columns)
    newY = pd.DataFrame(Y, columns=mlb.classes_)

    #plotting features vs labels

    new_df = pd.concat([newX, newY], axis=1)

    corr = new_df.corr()[mlb.classes_]
    mask = np.triu(np.ones_like(corr))

    heatmap = sns.heatmap(corr, mask=mask,vmin=-1, vmax=1, cmap='Blues', square=True, annot=False, cbar=True, linewidths=1,linecolor='Black')
    plt.savefig('heatmap_independentvsdependent.png', dpi=300, bbox_inches='tight')

def feature_correlations():
    df = tools.read_file('files', 'dupage_chinatown.csv')
    df_features = multilabel_classification.select_features(df)
    cols = df_features.columns.values

    # Calculate correlations between all features
    corr = df_features.corr()

    #Return a list with the given feature, the highest correlation with another feature, and that feature.
    col_name = 'visit_counts'
    path = os.environ['PYTHONPATH'] + os.path.sep + 'files' + os.path.sep + 'feature_correlations.csv'
    #doc = open(os.environ['PYTHONPATH'] + os.path.sep + 'files' + os.path.sep + 'correlations.txt','w')
    lst = []
    for c in cols:
        corrs = find_strongest_correlations(c, cols, corr)
        lst.append(corrs)
        #doc.write(str(corrs) + '\n')
    new_df = pd.DataFrame(lst,columns=['Feature','Most Positive Correlation','Most Positive Correlation-Feature','Most Negative Correlation','Most Negative Correlation-Feature'])
    print(new_df['Feature'])
    decode_features(new_df,columns=['Feature','Most Positive Correlation-Feature','Most Negative Correlation-Feature'])


    new_df.to_csv(path_or_buf=path)
    #doc.close()

def feature_actions_correlations():
    df = tools.read_file('files', 'dupage_chinatown.csv')
    df_features = multilabel_classification.select_features(df)
    feature_cols = df_features.columns

    labels = ['action_1', 'action_2', 'action_3']
    Y, mlb = tools.select_multilabels(df, labels)
    df_actions = pd.DataFrame(Y,columns=mlb.classes_)
    action_cols = mlb.classes_

    length = len(df_actions)
    ind = range(length)
    df_features['ind'] = ind
    df_actions['ind'] = ind
    merged = pd.merge(df_features,df_actions,on='ind')
    merged.set_index('ind', inplace=True)
    print(merged)

    # Calculate correlations between all features
    corr = merged.corr()
    print(corr)

    #Return a list with the given feature, the highest correlation with another feature, and that feature.

    path = os.environ['PYTHONPATH'] + os.path.sep + 'files' + os.path.sep + 'feature_action_correlations.csv'
    #doc = open(os.environ['PYTHONPATH'] + os.path.sep + 'files' + os.path.sep + 'correlations.txt','w')
    lst = []
    for c in action_cols:
        corrs = find_strongest_correlations(c, feature_cols, corr)
        lst.append(corrs)
        #doc.write(str(corrs) + '\n')
    new_df = pd.DataFrame(lst,columns=['Actions','Most Positive Correlation','Most Positive Correlation-Feature','Most Negative Correlation','Most Negative Correlation-Feature'])
    print(new_df['Actions'].values)
    decode_features(new_df,columns=['Most Positive Correlation-Feature','Most Negative Correlation-Feature'])
    decode_classes(new_df, classes='actions',columns=['Actions'])
    new_df.to_csv(path_or_buf=path)
    #doc.close()

#
def feature_barriers_correlations():
    df = tools.read_file('files', 'dupage_chinatown.csv')
    df_features = multilabel_classification.select_features(df)
    feature_cols = df_features.columns

    labels = ['barrier_1', 'barrier_2', 'barrier_3']
    Y, mlb = tools.select_multilabels(df, labels)
    df_barriers = pd.DataFrame(Y,columns=mlb.classes_)
    barrier_cols = mlb.classes_

    length = len(df_barriers)
    ind = range(length)
    df_features['ind'] = ind
    df_barriers['ind'] = ind
    merged = pd.merge(df_features,df_barriers,on='ind')
    merged.set_index('ind', inplace=True)
    print(merged)

    # Calculate correlations between all features
    corr = merged.corr()
    print(corr)

    #Return a list with the given feature, the highest correlation with another feature, and that feature.

    path = os.environ['PYTHONPATH'] + os.path.sep + 'files' + os.path.sep + 'feature_barrier_correlations.csv'
    #doc = open(os.environ['PYTHONPATH'] + os.path.sep + 'files' + os.path.sep + 'correlations.txt','w')
    lst = []
    for c in barrier_cols:
        corrs = find_strongest_correlations(c, feature_cols, corr)
        lst.append(corrs)
        #doc.write(str(corrs) + '\n')
    new_df = pd.DataFrame(lst,columns=['Barrier','Most Positive Correlation','Most Positive Correlation-Feature','Most Negative Correlation','Most Negative Correlation-Feature'])
    print(new_df['Barrier'].values)
    decode_features(new_df,columns=['Most Positive Correlation-Feature','Most Negative Correlation-Feature'])
    decode_classes(new_df,classes='barriers',columns=['Barrier'])
    new_df.to_csv(path_or_buf=path)
    #doc.close()

def decode_classes(df, classes='None',columns=[]):
    barrier_conversions = {'CC1':' Unaware of Eligibility', 'CC3':'Believes They are Ineligible',
                   'CM':'Case Management','CP1':' Does not understand cancer',
                   'CP2':' Does not understand testing','CP4':'Does not understand instructions',
                   'CP5':'Does not understand forms','F1':'Fear of Unknown Results',
                   'F2':'Fear of Testing','F3':'Fear of Treatment','FC1':'Cannot find eldercare',
                   'FP1':'Cannot afford housing','FP2':'Cannot afford food',
                   'FP3':'Cannot afford utilities','FP4':'Cannot afford clothing',
                   'H1':'Homeless','H2':'Inconsistent housing','I1':'None',
                   'I2':'Expensive copay/deductable','I3':'Insufficient discounts/payment plan',
                   'I5':'Anxious of bills','I6':'Overwhelmed by paperwork',
                   'I7':'Misunderstands coverage','L1':'Illiterate/Low literacy/Learning disability',
                   'L2':'Does not understand medical terminology','LI1':'Uncomfortable with English',
                   'LI2':'Staff do not speak native language','MM1':'Comorbidities',
                   'MM3':'Testing painful','MM4':'Side effects',
                   'MM5':'Stress','MM6':'Depressed/Unmotivated',
                   'MM7':'Mental illness','MM8':'Substance abuse',
                   'NB1':'Navigator Barrier 1', 'NB4':'Navigator Barrier 4','NB5':'Navigator Barrier 5',
                   'None':'No Barrier','O1':'Out of country/region','Other':'Other/No relevant tag',
                   'PB1':'Beliefs:Results irrelevant/Trust higher power','RD7':'RD7','SP1':'No near family support',
                   'SP3':'No reliable friends support','SP4':'Abusive/Violent home','SS1':'No phone/Disconnected',
                   'SS2':'Difficulties during phone call', 'SS6':'Inconvenient hours','T1':'Lack of public transportation',
                   'T2':'Fare/Fuel unaffordable','T3':'No available drivers','W1':'Concerns of losing job/benefits'}
    action_conversions = {'AC1':'To health service','AC2':'To other service',
                          'AR1':'Transporation','AR2':'Interpreter',
                          'AR5':'Financial assistance','AR8':'Public assistance program enrollment',
                          'E1':'Verbal explanation','E3':'Print/Audio-Visual material',
                          'E4':'Translation for patient','None':'No intervention taken/needed','Other':'Other intervention/No relevant code',
                          'PN1':'Verification/Information gathering','PN3':'Educate staff of patient special needs',
                          'R1':'Aid in paperwork completion',
                          'R2':'Request/Organize health records','R3':'Provide documents to providers',
                          'R4':'Consent/Exit patient','RD1':'Call patient',
                          'RD2':'Referral for social services agency','RD3':'Referral for health care',
                          'RD4':'Contact social service agency','RD5':'Contact family',
                          'RD6':'Contact other support systems','RD7':'Contact health care providers',
                          'S1':'Emotional support/Active listening','S2':'Motivational counseling',
                          'SA1':'Schedule/Reschedule appointments','SA2':'Contact patient/Reminder',
                          'SA3':'Mail letter to patient'}
    if classes == 'None':
        return
    elif classes == 'barriers':
        if len(columns) == 0:
            print('No Columns to Decode Given')
            return
        for col in columns:
            for index, c in enumerate(df[col].values):
                if c in barrier_conversions.keys():
                    df.at[index,col] = barrier_conversions[c]
    elif classes == 'actions':
        if len(columns) == 0:
            print('No Columns to Decode Given')
            return
        for col in columns:
            for index, c in enumerate(df[col].values):
                if c in action_conversions.keys():
                    df.at[index,col] = action_conversions[c]


def decode_features(df,columns=[]):
    if len(columns) == 0:
        print('No Columns to Decode Given')
        return
    conversions = {
        'PDAGE': 'Age', 'visit_counts': 'Visits', 'PDHSIZE': 'Household Size', 'PDLANG_1': 'English Primary Language',
        'PDLANG_10': 'Other Primary Language', 'PDLANG_2': 'Spanish Primary Language',
        'PDLANG_6': 'Albanian Primary Language', 'PDLANG_None': 'Primary Language Not Chosen',
        'PDZIP_60101': '60101', 'PDZIP_60103': '60103', 'PDZIP_60106': '60106',
        'PDZIP_60108': '60108', 'PDZIP_60126': '60126', 'PDZIP_60133': '60133',
        'PDZIP_60134': '60134', 'PDZIP_60137': '60137', 'PDZIP_60139': '60139',
        'PDZIP_60143': '60143', 'PDZIP_60148': '60148', 'PDZIP_60166': '60166',
        'PDZIP_60172': '60172', 'PDZIP_60181': '60181', 'PDZIP_60185': '60185',
        'PDZIP_60186': '60186', 'PDZIP_60187': '60187', 'PDZIP_60188': '60188',
        'PDZIP_60189': '60189', 'PDZIP_60190': '60190', 'PDZIP_60191': '60191',
        'PDZIP_60440': '60440', 'PDZIP_60502': '60502', 'PDZIP_60504': '60504',
        'PDZIP_60514': '60514', 'PDZIP_60515': '60515', 'PDZIP_60516': '60516',
        'PDZIP_60517': '60517', 'PDZIP_60519': '60519', 'PDZIP_60521': '60521',
        'PDZIP_60527': '60527', 'PDZIP_60532': '60532', 'PDZIP_60540': '60540',
        'PDZIP_60555': '60555', 'PDZIP_60559': '60559', 'PDZIP_60561': '60561',
        'PDZIP_60563': '60563', 'PDZIP_60564': '60564', 'PDZIP_60565': '60565',
        'PDZIP_60605': '60605', 'PDZIP_60607': '60607', 'PDZIP_60608': '60608',
        'PDZIP_60609': '60609', 'PDZIP_60616': '60616', 'PDZIP_60632': '60632',
        'PDZIP_60635': '60635', 'PDZIP_60660': '60660', 'PDZIP_None': 'Zipcode Not Given',
        'PDBIRTH_AF': 'Afghanistan ', 'PDBIRTH_AL': 'Albania',
        'PDBIRTH_AO': 'Angola', 'PDBIRTH_AR': 'Argentina',
        'PDBIRTH_BD': 'Bangladesh', 'PDBIRTH_BO': 'Bolivia',
        'PDBIRTH_CL': 'Chile', 'PDBIRTH_CN': 'China',
        'PDBIRTH_CO': 'Columbia', 'PDBIRTH_CU': 'Cuba',
        'PDBIRTH_DE': 'Germany', 'PDBIRTH_EC': 'Ecuador',
        'PDBIRTH_GR': 'Greece', 'PDBIRTH_GT': 'Guatemala',
        'PDBIRTH_HN': 'Honduras', 'PDBIRTH_IN': 'India',
        'PDBIRTH_IR': 'Iran', 'PDBIRTH_IT': 'Italy',
        'PDBIRTH_JM': 'Jamaica', 'PDBIRTH_JO': 'Jordan',
        'PDBIRTH_LB': 'Lebanon', 'PDBIRTH_LT': 'Lithuania',
        'PDBIRTH_MX': 'Mexico', 'PDBIRTH_None': 'Not Given',
        'PDBIRTH_PE': 'Peru', 'PDBIRTH_PH': 'Philippines',
        'PDBIRTH_PL': 'Poland', 'PDBIRTH_PR': 'Puerto Rico',
        'PDBIRTH_RU': 'Russia', 'PDBIRTH_SD': 'Sudan',
        'PDBIRTH_SV': 'El Salvador', 'PDBIRTH_UK': 'United Kingdom',
        'PDBIRTH_US': 'USA', 'PDBIRTH_VN': 'Vietnam',
        'Mstat_1': 'Single/Never Married', 'Mstat_2': 'Married/Living as Married',
        'Mstat_3': 'Divorced/Separated', 'Mstat_4': 'Widowed', 'Mstat_98': 'Marital Status Not Given',
        'Edu_1': '8th Grade or Less', 'Edu_2': 'Some High School', 'Edu_3': 'High School Diploma/Equivalent',
        'Edu_4': 'Some College/Vocational After High School', 'Edu_5': 'Associate Degree',
        'Edu_6': 'College Graduate', 'Edu_7': 'Graduate or Professional Degree',
        'Edu_98': 'Education Not Given', 'income_1': '<$10,000', 'income_2': '$10,000 to $19,999',
        'income_3': '$20,000 to $29,999', 'income_4': '$30,000 to $39,999',
        'income_5': '$40,000 to $49,999', 'income_6': '$50,000<', 'income_98': 'Income Not Given',
        'Emp_Housewife': 'Employment: Homemaker', 'Emp_None': 'Employment Not Given',
        'Emp_Other': 'Employment: Other', 'Emp_Retired': 'Employment: Retired',
        'Emp_Student': 'Employment: Student', 'Emp_Unemployed': 'Employment: Unemployed',
        'Emp_Working': 'Employment: Working'}
    for col in columns:
        for index, c in enumerate(df[col].values):
            df.at[index,col] = conversions[c]


def find_strongest_correlations(col_name, cols, corr):
    corrs = []
    m = -1
    j = 0
    vals = corr[col_name][cols].values
    corrs.append(col_name)

    #Find max correlation and corresponding feature and append values to the output list.
    maxdex = np.where(corr.columns.values == col_name)[0]
    temp = -1.0
    for index, j in enumerate(vals):
        if index == maxdex:
            continue
        if j > temp:
            temp = j
            m = index
    corrs.append(temp)
    corrs.append(cols[m])

    #Find min correlation and append value and feature to output list.
    mindex = min(vals)
    corrs.append(mindex)
    arr = np.where(vals == mindex)
    corrs.append(cols[arr[0]][0])

    return corrs

if __name__ == '__main__':
    feature_correlations()
    feature_barriers_correlations()
    feature_actions_correlations()