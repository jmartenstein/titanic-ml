import pandas as pd
import re

### Functions ###

def split_last_name( row ):
    l_subs = row['Name'].split(r', ')
    return l_subs

def split_title( row ):
    l_subs = row[1].split(r'. ')
    if len(l_subs) > 2:
        l_subs[1] = f"{l_subs[1]} {l_subs[2]}"
        l_subs.pop()
    return l_subs


### Main ###

filename = "test"
df = pd.read_csv(f"../data/kaggle/{filename}.csv")

temp1 = df.apply( split_last_name, axis=1, result_type='expand' )
df["LastName"] = temp1[0]

temp2 = temp1.apply( split_title, axis=1, result_type='expand' )
df["Title"] = temp2[0]

df.to_csv(f"../data/kaggle/{filename}_expanded.csv", index=False)
