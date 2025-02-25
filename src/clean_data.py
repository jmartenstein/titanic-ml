import pandas as pd
import numpy as np

import argparse
import re

from sklearn  import preprocessing, impute
from datetime import datetime

### Constants ###

DATA_DIR = "../data/kaggle"


### Functions ###

def get_outfile_name( in_file ):

    f_split = in_file.split(".")
    now = datetime.now()

    s_date = now.strftime("%Y%m%d")
    s_time = now.strftime("%H%M%S")

    return f"{f_split[0]}.clean.{s_date}.{s_time}.{f_split[1]}"

def split_last_name( row ):
    l_subs = row['Name'].split(r', ')
    return l_subs

def split_title( row ):
    l_subs = row[1].split(r'. ')
    if len(l_subs) > 2:
        l_subs[1] = f"{l_subs[1]} {l_subs[2]}"
        l_subs.pop()
    return l_subs

def filter_age( row, n_filter ):

    n_result = 0
    n_age = row["Age"]

    if ~np.isnan(n_age):
        if n_age < n_filter:
            n_result = 1

    return n_result


def get_name_and_title( df ):

    temp1 = df.apply( split_last_name, axis=1, result_type='expand' )
    df["LastName"] = temp1[0]

    temp2 = temp1.apply( split_title, axis=1, result_type='expand' )
    df["Title"] = temp2[0]

    return df

def print_subset( df ):

    df.sort_values(['GroupSize', 'LastName'], ascending=False, inplace=True)
    print(df.columns)
    print(df[ ["Name", "LastName", "Pclass", "FarePerPerson", "Age", "AgeImputed", "Survived" ]].to_string())

    return True

def get_p3_or_dead_title(row):
    l_dead_titles = [ 'Capt', 'Rev', 'Mr' ]

    s_title = row['Title']
    b_pclass3 = row['Pclass3']

    if ( s_title in l_dead_titles ) or ( b_pclass3 == 1 ):
        return 1
    else:
        return 0

def get_other_titles( row ):

    l_titles = [ 'Mr', 'Mrs', 'Miss', 'Master' ]
    s_title = row['TitleGrouped']

    if (not s_title in l_titles):
        if row['Sex'] == 'male':
            return 'OtherMale'
        else:
            return 'OtherFemale'
    else:
        return s_title

def get_title_dummies( df ):

    df_temp = pd.DataFrame(df[["PassengerId", "TitleGrouped", "Sex"]])

    df_titles = pd.get_dummies(df_temp['TitleGrouped'])
    df_titles["PassengerId"] = df_temp['PassengerId']
    df_titles = df_titles.map(lambda x: int(x))

    return df_titles

def scaler_fit_transform( scaler, df ):

    l_transformed = scaler.fit_transform(df)
    return l_transformed[:,0].round(4)

### Main ###

parser = argparse.ArgumentParser( description='Pre-process data from Titanic dataset' )
parser.add_argument('-w', '--write', action='store_true', help='Write processed output to files')

args = vars( parser.parse_args() )

# load train data
df_train = pd.read_csv(f"../data/kaggle/train.csv")


# load test data with a Survived column to encode data
df_test  = pd.read_csv(f"../data/kaggle/test.csv")
df_test["Survived"] = np.nan

df_full = pd.concat([df_train, df_test])

# preprocess categorical data (sex and title) into ordinal values
enc = preprocessing.OrdinalEncoder(max_categories=5)
df_full = get_name_and_title( df_full )
df_full["TitleOrd"] = enc.fit_transform(df_full[["Title"]])
df_full["SexOrd"] = enc.fit_transform(df_full[["Sex"]])

df_full['TitleGrouped'] = df_full['Title']
df_full['TitleGrouped'] = df_full.apply( get_other_titles, axis=1 )

# create dummy variables / categories for the 5 title groupings
df_titles = get_title_dummies( df_full )
df_full = pd.merge(df_full, df_titles, on="PassengerId")

df_full["IsMale"]        = df_full["Sex"].apply( lambda v: 1 if v == "male" else 0)
df_full["IsChild"]       = df_full["Age"].apply( lambda v: 1 if v <= 16 else 0)
df_full["IsYoungChild"]  = df_full["Age"].apply( lambda v: 1 if v <= 8 else 0)
df_full["GroupSize"]     = df_full["SibSp"] + df_full["Parch"] + 1
df_full["LargeGroup"]    = df_full["GroupSize"].apply( lambda v: 1 if v >= 5 else 0)
df_full["IsAlone"]       = df_full["GroupSize"].apply( lambda v: 1 if v == 1 else 0 )
df_full["Fare"]          = df_full["Fare"].apply( lambda v: 0 if np.isnan(v) else v )
df_full["FarePerPerson"] = round((df_full["Fare"] / df_full["GroupSize"]),4)
df_full["Pclass3"]       = df_full["Pclass"].apply( lambda v: 1 if v == 3 else 0 )

df_full["P3orDeadTitle"] = df_full.apply( get_p3_or_dead_title, axis=1 )

imputer = impute.KNNImputer(n_neighbors=5)
l_imputed = imputer.fit_transform(df_full[[ "Age", "IsMale", "Mr", "Mrs", "Miss", "Master", "OtherMale", "OtherFemale", "Pclass" ]])

df_full["AgeImputed"] = l_imputed[:,0].round(4)

# normalize the age and fare per person features, to reduce impact of outliers
robust = preprocessing.RobustScaler()
minmax = preprocessing.MinMaxScaler()

df_full["AgeRobust"] = scaler_fit_transform( robust, df_full[["AgeImputed"]] )
df_full["AgeMinMax"] = scaler_fit_transform( minmax, df_full[["AgeImputed"]] )

df_full["FppRobust"] = scaler_fit_transform( robust, df_full[["FarePerPerson"]] )
df_full["FppMinMax"] = scaler_fit_transform( minmax, df_full[["FarePerPerson"]] )

#print_subset(df_full[ (df_full["Age"].isna() ) ].copy())

# prep the cleaned training data to write
df_train_clean = df_full[ df_full[ "Survived" ].notna() ].copy()

# prep the cleaned test data to write
df_test_clean = df_full[ df_full[ "Survived" ].isna() ].copy()
df_test_clean.drop( "Survived", axis=1, inplace=True )

if args["write"]:

    train_outfile = get_outfile_name("train.csv")
    print(f"write train shape: {df_train_clean.shape} to file: {train_outfile}")
    df_train_clean.to_csv(f"../data/kaggle/{train_outfile}", index=False)

    test_outfile = get_outfile_name("test.csv")
    print(f"write test shape:  {df_test_clean.shape} to file: {test_outfile}")
    df_test_clean.to_csv(f"../data/kaggle/{test_outfile}", index=False)

else:

    print("No output written to files")
    print(f"  train shape: {df_train_clean.shape}")
    print(f"  test shape:  {df_test_clean.shape}")
