import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.metrics as ms
import statistics as st

from sklearn import svm, preprocessing, ensemble
from sklearn.model_selection import train_test_split, \
                                    KFold, \
                                    cross_val_score

### FUNCTIONS ###

def print_accuracy_scores( model, X, y, num_folds ):

    kf = KFold(n_splits=num_folds)

    mean_accuracy_scores = cross_val_score( model, X, y, cv=kf)
    mean_accuracy1 = model.score(X, y)
    mean_accuracy2 = st.mean(mean_accuracy_scores)

    print(f"accuracy from score: {mean_accuracy1}")
    print(f"acc score {num_folds} folds:   {mean_accuracy2}")
    print(f"  each: {mean_accuracy_scores}")
    print()

    return True

def print_feature_importance( model, colnames ):

    importances = model.feature_importances_
    feature_imp_df = pd.DataFrame(
        {'Feature': colnames, 'Gini Importance': importances}).sort_values(
            'Gini Importance', ascending=False)
    print(feature_imp_df)
    print()

    return True

def plot_roc_curve( model, X, y ):

    probs = model.predict_proba(X)
    preds = probs[:,1]

    fpr, tpr, threshold = ms.roc_curve(y, preds)
    roc_auc = ms.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')

    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
    plt.plot([0, 1], [0, 1],'r--')

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    plt.legend(loc = 'lower right')
    plt.show()

    return True


### MAIN ###

datestamp = "20250129.150949"
df = pd.read_csv(f"../data/kaggle/train.clean.{datestamp}.csv")

#x_colnames = [ "Pclass", "TitleOrd", "GroupSize", "SexOrd", "IsChild", "IsYoungChild", "Parch", "SibSp", "AgeImputed", "Fare", "FarePerPerson" ]
x_colnames = [ "SexOrd", "FarePerPerson", "AgeImputed", "GroupSize", "Pclass" ]
y_colname = [ "Survived" ]

X = df[ x_colnames ]
y = df[ y_colname ].values.ravel()

#corr_matrix = pd.DataFrame(X, columns=x_colnames).corr()
#sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu')
#plt.show()


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3)
print(f"training data shape: {X_train.shape}")
print(f"test data shape:     {X_test.shape}")

classifier = ensemble.GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=42)
#classifier = ensemble.RandomForestClassifier(n_estimators=120, random_state=42)

model = classifier.fit(X_train, y_train)

print_accuracy_scores(model, X_test, y_test, 5)
print_feature_importance(model, x_colnames)
plot_roc_curve( model, X_test, y_test )

df_test = pd.read_csv(f"../data/kaggle/test.clean.{datestamp}.csv")
y_pred = model.predict_proba(df_test[ x_colnames ])
df_test["SurvivedProbability"] = y_pred[:,1]
df_test["Survived"] = (y_pred[:,1] > 0.80).astype(int)

print(y_pred[-5:])
print(df_test[["PassengerId", "Sex", "SurvivedProbability"]].tail(10))

df_test[["PassengerId", "Survived"]].to_csv(f"../data/kaggle/submit.{datestamp}.csv", index=False)
