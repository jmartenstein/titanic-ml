import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.metrics as ms
import statistics as st
import numpy as np

from sklearn import svm, preprocessing, ensemble
from sklearn.model_selection import TunedThresholdClassifierCV, \
                                    RandomizedSearchCV, \
                                    train_test_split, \
                                    KFold, \
                                    cross_val_score

### FUNCTIONS ###

def print_crossval_scores( model, X, y, num_folds ):

    kf = KFold(n_splits=num_folds)

    s_scoring = "f1"
    l_raw_scores = cross_val_score( model, X, y, scoring=s_scoring, cv=kf)
    l_scores = [ round(i, 2) for i in l_raw_scores ]

    mean_score = round(st.mean(l_scores), 3)
    std_score = round(st.stdev(l_scores), 3)

    #print(f"accuracy from score: {mean_accuracy1}")
    print(f"{num_folds}-fold {s_scoring} mean: {mean_score}, stdev: {std_score}")
    print(f"  each: {l_scores}")
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
df_train = pd.read_csv(f"../data/kaggle/train.clean.{datestamp}.csv")
df_test = pd.read_csv(f"../data/kaggle/test.clean.{datestamp}.csv")

#x_colnames = [ "Pclass", "TitleOrd", "GroupSize", "SexOrd", "IsChild", "IsYoungChild", "Parch", "SibSp", "AgeImputed", "Fare", "FarePerPerson" ]
x_colnames = [ "SexOrd", "FarePerPerson", "AgeImputed" ]
y_colname = [ "Survived" ]

X = df_train[ x_colnames ]
y = df_train[ y_colname ].values.ravel()

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.4)
print("Data Shapes")
print(f"  train: {X_train.shape}")
print(f"  test:  {X_test.shape}")
print()

param_dist = {
    'learning_rate': np.arange(0.01, 0.1, 0.01),
    'n_estimators': [50, 75, 100, 125, 150, 175, 200],
    'max_depth': [2, 3, 4, 5, 6],
    'subsample': np.arange(0.2, 0.8, 0.1),
    'max_features': [ 2 ],
    'min_samples_split': [ 30 ]
}

gb_classifier = ensemble.GradientBoostingClassifier()
random_search = RandomizedSearchCV( estimator=gb_classifier,
                                    param_distributions=param_dist,
                                    n_iter=50,
                                    cv=5,
                                    scoring='f1',
                                    n_jobs=-1 )

#classifier = ensemble.GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=42)
#classifier = ensemble.RandomForestClassifier(n_estimators=120, random_state=42)
#model = classifier.fit(X_train, y_train)

random_search.fit(X_train, y_train)
best_params = random_search.best_params_
model = random_search.best_estimator_

print(f"Best Params: {best_params}")
print()

print_crossval_scores(model, X_test, y_test, 5)

threshold_tuner = TunedThresholdClassifierCV(
    gb_classifier, scoring="f1", cv=5).fit(X_train, y_train)
threshold = threshold_tuner.best_threshold_

print(f"Threshold classfier score: {threshold_tuner.best_score_}")
print()

y_test_preds_proba = model.predict_proba(X_test)
y_test_preds = (y_test_preds_proba[:,1] > threshold).astype(int)

tn, fp, fn, tp = ms.confusion_matrix(y_test, y_test_preds).ravel()
print(f"Confusion Matrix, at Threshold {threshold:.3f}")
print(f"  tn {tn:<3}  fp {fp:<3}")
print(f"  fn {fn:<3}  tp {tp:<3}")
print()

y_proba = model.predict_proba(df_test[x_colnames])
y_preds = (y_proba[:,1] > threshold).astype(int)

df_test["SurvivedProbability"] = y_proba[:,1]
df_test["Survived"] = y_preds

print_feature_importance(model, x_colnames)
#plot_roc_curve( model, X_test, y_test )

df_test[["PassengerId", "Survived"]].to_csv(f"../data/kaggle/submit.{datestamp}.csv", index=False)
