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

#
def feature_barriers_correlations():
    df = tools.read_file('files', 'dupage_chinatown.csv')
    df_features = multilabel_classification.select_features(df)
    labels = ['barrier_1', 'barrier_2', 'barrier_3']
    Y, mlb = tools.select_multilabels(df, labels)
    df_barriers = pd.DataFrame(Y,columns=mlb.classes_)
    print(df_barriers)
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

def decode_features(df,columns=[]):
    if len(columns) == 0:
        print('No Columns to Decode Given')
        return
    lst = {
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
        'PDBIRTH_AF': 'Birth Country: Afghanistan ', 'PDBIRTH_AL': 'Birth Country: Albania',
        'PDBIRTH_AO': 'Birth Country: Angola', 'PDBIRTH_AR': 'Birth Country: Argentina',
        'PDBIRTH_BD': 'Birth Country: Bangladesh', 'PDBIRTH_BO': 'Bolivia',
        'PDBIRTH_CL': 'Birth Country: Chile', 'PDBIRTH_CN': 'Birth Country: China',
        'PDBIRTH_CO': 'Birth Country: Columbia', 'PDBIRTH_CU': 'Birth Country: Cuba',
        'PDBIRTH_DE': 'Birth Country: Germany', 'PDBIRTH_EC': 'Birth Country: Ecuador',
        'PDBIRTH_GR': 'Birth Country: Greece', 'PDBIRTH_GT': 'Birth Country: Guatemala',
        'PDBIRTH_HN': 'Birth Country: Honduras', 'PDBIRTH_IN': 'Birth Country: India',
        'PDBIRTH_IR': 'Birth Country: Iran', 'PDBIRTH_IT': 'Birth Country: Italy',
        'PDBIRTH_JM': 'Birth Country: Jamaica', 'PDBIRTH_JO': 'Birth Country: Jordan',
        'PDBIRTH_LB': 'Birth Country: Lebanon', 'PDBIRTH_LT': 'Birth Country: Lithuania',
        'PDBIRTH_MX': 'Birth Country: Mexico', 'PDBIRTH_None': 'Birth Country: Not Given',
        'PDBIRTH_PE': 'Birth Country: Peru', 'PDBIRTH_PH': 'Birth Country: Philippines',
        'PDBIRTH_PL': 'Birth Country: Poland', 'PDBIRTH_PR': 'Birth Country: Puerto Rico',
        'PDBIRTH_RU': 'Birth Country: Russia', 'PDBIRTH_SD': 'Birth Country: Sudan',
        'PDBIRTH_SV': 'Birth Country: El Salvador', 'PDBIRTH_UK': 'Birth Country: United Kingdom',
        'PDBIRTH_US': 'Birth Country: USA', 'PDBIRTH_VN': 'Birth Country: Vietnam',
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
            df.at[index,col] = lst[c]


def find_strongest_correlations(col_name, cols, corr):
    corrs = []
    m = -1
    j = 0
    vals = corr[col_name].values
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
    #feature_correlations()
    feature_barriers_correlations()