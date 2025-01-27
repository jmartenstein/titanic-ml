import pandas as pd
import numpy as np

def is_surviving_title( row ):
    nonsurviving_titles = ['Capt', 'Don', 'Rev', 'Jonkheer', 'Mr', 'Dr', 'Major', 'Col']

    if row['Title'] in nonsurviving_titles:
        return 0
    else:
        return 1

df = pd.read_csv("../data/kaggle/test_expanded.csv")
df['IsSurvivalTitle'] = df.apply(is_surviving_title, axis=1)

df_out = df[[ "PassengerId", "IsSurvivalTitle" ]]
df_out.columns = ["PassengerId", "Survived"]
#df_out.rename(columns = {"IsSurvivalTitle": "Survived"})

#print(df_out)
df_out.to_csv("../data/kaggle/title_submission.csv", index=False)
