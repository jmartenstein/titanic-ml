import pandas as pd
import numpy as np

import argparse
import re
import math

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

def split_cabin_deck( row ):

    if type(row) is float:
        return ""
    else:
        # split the letters into a list
        l_letters = re.findall( "[a-zA-Z]+", row )

        # check if all of the letters are the same; in rare cases where they
        # are not the same, the first character is "F", so we're choosing the
        # second
        if all(x == l_letters[0] for x in l_letters):
            return l_letters[0]
        else:
            return l_letters[1]

def split_cabin_room( row ):

    if type(row) is float:
        return ""
    else:
        # split the letters into a list
        l_numbers = re.findall( "[0-9]+", row )

        if len(l_numbers) == 0:
            return ""
        else:
            median = math.ceil((len(l_numbers) - 1) / 2)
            if all(x == l_numbers[0] for x in l_numbers):
                return int(l_numbers[0])
            else:
                return int(l_numbers[median])

def print_subset( df ):

    #df.sort_values(['GroupSize', 'LastName'], ascending=False, inplace=True)
    print(df.columns)
    print(df[ ["LastName", "GroupSurvived", "Title", "TitleOrd", "Survived", "Pclass", "Embarked", "Cabin", "Ticket" ]].to_string())

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

def get_column_dummies( df, s_column ):

    df_temp = pd.DataFrame(df[["PassengerId", s_column ]])

    df_titles = pd.get_dummies(df_temp[s_column])
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
title_enc = preprocessing.OrdinalEncoder(
    categories=[['Mr', 'OtherMale', 'Master', 'Miss', 'Mrs', 'OtherFemale']])
enc = preprocessing.OrdinalEncoder()
#df_full = get_name_and_title( df_full )

df_temp_name = df_full.apply( split_last_name, axis=1, result_type='expand' )
df_temp_title = df_temp_name.apply( split_title, axis=1, result_type='expand' )

df_full["LastName"] = df_temp_name[0]
df_full["Title"] = df_temp_title[0]

df_full['TitleGrouped'] = df_full['Title']
df_full['TitleGrouped'] = df_full.apply( get_other_titles, axis=1 )

df_full["LastNameOrd"] = enc.fit_transform(df_full[["LastName"]])
df_full["TitleOrd"] = title_enc.fit_transform(df_full[["TitleGrouped"]])
df_full["SexOrd"] = enc.fit_transform(df_full[["Sex"]])

# create dummy variables / categories for the 5 title groupings
df_titles = get_column_dummies( df_full, 'TitleGrouped' )
df_full = pd.merge(df_full, df_titles, on="PassengerId")

x_colname = "Ticket"
xcol_frequency = df_full[[x_colname]].value_counts()
df_frequency = xcol_frequency.reset_index()
df_frequency.columns = [x_colname, "TicketFrequency"]

xcol_survived = df_full.groupby([x_colname])["Survived"].sum()
df_survived = xcol_survived.reset_index()
df_survived.columns = [x_colname, "SurvivorCount"]

df_survived = df_survived.merge(df_frequency, on="Ticket")
df_survived["GroupSurvived"] = df_survived["SurvivorCount"].apply( lambda v: 1 if v > 1 else 0 )

df_full = df_full.merge(df_survived, on="Ticket")

df_full["IsMale"]        = df_full["Sex"].apply( lambda v: 1 if v == "male" else 0)
df_full["IsChild"]       = df_full["Age"].apply( lambda v: 1 if v <= 16 else 0)
df_full["IsYoungChild"]  = df_full["Age"].apply( lambda v: 1 if v <= 8 else 0)
df_full["GroupSize"]     = df_full["SibSp"] + df_full["Parch"] + 1
df_full["LargeGroup"]    = df_full["GroupSize"].apply( lambda v: 1 if v >= 5 else 0)
df_full["SmallGroup"]    = df_full["GroupSize"].apply( lambda v: 1 if (v > 1 and v < 5) else 0 )
df_full["IsAlone"]       = df_full["GroupSize"].apply( lambda v: 1 if v == 1 else 0 )
df_full["Fare"]          = df_full["Fare"].apply( lambda v: 0 if np.isnan(v) else v )
df_full["FarePerPerson"] = round((df_full["Fare"] / df_full["GroupSize"]),4)

df_full["Pclass1"]       = df_full["Pclass"].apply( lambda v: 1 if v == 1 else 0 )
df_full["Pclass2"]       = df_full["Pclass"].apply( lambda v: 1 if v == 2 else 0 )
df_full["Pclass3"]       = df_full["Pclass"].apply( lambda v: 1 if v == 3 else 0 )

df_full["P3orDeadTitle"] = df_full.apply( get_p3_or_dead_title, axis=1 )

df_full["HasCabin"]      = df_full["Cabin"].apply( lambda v: 0 if pd.isna(v) else 1 )
df_full["CabinDeck"]     = df_full["Cabin"].apply( split_cabin_deck )
df_full["CabinRoom"]     = df_full["Cabin"].apply( split_cabin_room )

df_full["CabinDeck"].fillna(value="X", inplace=True)
df_full["CabinOrd"] = enc.fit_transform(df_full[["CabinDeck"]])

# create dummy variables / categories for the 5 title groupings
df_cabins = get_column_dummies( df_full, 'CabinDeck' )
for i in ["A", "B", "C", "D", "E", "F", "G", "T"]:
    df_full["Cabin" + i ] = df_cabins[i]
#df_full = pd.merge(df_full, df_cabins, on="PassengerId")

# create dummy variables / categories for the 5 title groupings
df_full["Embarked"].fillna(value="S", inplace=True)
df_embarked = get_column_dummies( df_full, 'Embarked' )
for i in ["C", "Q", "S" ]:
    df_full["Embark" + i ] = df_embarked[i]
df_full["EmbarkOrd"] = enc.fit_transform(df_full[["Embarked"]])

imputer = impute.KNNImputer(n_neighbors=5)
l_imputed = imputer.fit_transform(df_full[[ "Age", "Mr", "Mrs", "Miss", "Master", "OtherMale", "OtherFemale", "Pclass" ]])

df_full["AgeImputed"] = l_imputed[:,0].round(4)

# normalize the age and fare per person features, to reduce impact of outliers
robust = preprocessing.RobustScaler()
minmax = preprocessing.MinMaxScaler()

df_full["AgeRobust"] = scaler_fit_transform( robust, df_full[["AgeImputed"]] )
df_full["AgeMinMax"] = scaler_fit_transform( minmax, df_full[["AgeImputed"]] )

df_full["FppRobust"] = scaler_fit_transform( robust, df_full[["FarePerPerson"]] )
df_full["FppMinMax"] = scaler_fit_transform( minmax, df_full[["FarePerPerson"]] )

print_subset(df_full[ df_full["GroupSize"] > 5].head(40))

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
