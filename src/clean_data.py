import pandas as pd
import numpy as np

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

def get_title_dummies( df ):

    l_titles = [ 'Mr', 'Mrs', 'Miss', 'Master' ]

    df_temp = pd.DataFrame(df[["PassengerId", "Title"]])

    df_temp['TitleReduced'] = df_temp['Title']
    df_temp['TitleReduced'] = df_temp['TitleReduced'].apply(
        lambda v: 'Other' if not v in l_titles else v )

    df_titles = pd.get_dummies(df_temp['TitleReduced'])
    df_titles["PassengerId"] = df_temp['PassengerId']
    df_titles = df_titles.map(lambda x: int(x))

    return df_titles

### Main ###

# load train data
df_train = pd.read_csv(f"../data/kaggle/train.csv")

# load test data with a Survived column to encode data
df_test  = pd.read_csv(f"../data/kaggle/test.csv")
df_test["Survived"] = np.nan

df_full = pd.concat([df_train, df_test])

enc = preprocessing.OrdinalEncoder(max_categories=5)
df_full = get_name_and_title( df_full )
df_full["TitleOrd"] = enc.fit_transform(df_full[["Title"]])
df_full["SexOrd"] = enc.fit_transform(df_full[["Sex"]])

df_titles = get_title_dummies( df_full )
df_full = pd.merge(df_full, df_titles, on="PassengerId")

df_full["IsMale"]        = df_full["Sex"].apply( lambda v: 1 if v == "male" else 0)
df_full["IsChild"]       = df_full["Age"].apply( lambda v: 1 if v <= 16 else 0)
df_full["IsYoungChild"]  = df_full["Age"].apply( lambda v: 1 if v <= 8 else 0)
df_full["GroupSize"]     = df_full["SibSp"] + df_full["Parch"] + 1
df_full["IsAlone"]       = df_full["GroupSize"].apply( lambda v: 1 if v == 1 else 0 )
df_full["Fare"]          = df_full["Fare"].apply( lambda v: 0 if np.isnan(v) else v )
df_full["FarePerPerson"] = round((df_full["Fare"] / df_full["GroupSize"]),4)

imputer = impute.KNNImputer(n_neighbors=5)
l_imputed = imputer.fit_transform(df_full[[ "Age", "SexOrd", "TitleOrd", "Pclass", "FarePerPerson", "GroupSize" ]])
df_full["AgeImputed"] = l_imputed[:,0].round(4)

df_sub = df_full[ (df_full["Age"].isna()) ].copy()
#df_sub.sort_values(['GroupSize', 'LastName'], ascending=False, inplace=True)
#print(df_sub.columns)
#print(df_sub[ ["Name", "LastName", "Pclass", "FarePerPerson", "Age", "AgeImputed", "Survived" ]].to_string())

# prep the cleaned training data to write
df_train_clean = df_full[ df_full[ "Survived" ].notna() ].copy()

# prep the cleaned test data to write
df_test_clean = df_full[ df_full[ "Survived" ].isna() ].copy()
df_test_clean.drop( "Survived", axis=1, inplace=True )

train_outfile = get_outfile_name("train.csv")
print(f"write train shape: {df_train_clean.shape} to file: {train_outfile}")
df_train_clean.to_csv(f"../data/kaggle/{train_outfile}", index=False)

test_outfile = get_outfile_name("test.csv")
print(f"write test shape:  {df_test_clean.shape} to file: {test_outfile}")
df_test_clean.to_csv(f"../data/kaggle/{test_outfile}", index=False)

