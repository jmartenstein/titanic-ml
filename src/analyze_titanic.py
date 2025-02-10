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

def print_y_scores( y_true, y_pred ):

    tn, fp, fn, tp = ms.confusion_matrix(y_true, y_pred).ravel()
    print(f"Confusion Matrix: tn {tn}, fp {fp}, fn {fn}, tp {tp:<3}")

    bal_acc   = round( ms.balanced_accuracy_score( y_true, y_pred ), 3 )
    precision = round( ms.precision_score( y_true, y_pred ), 3 )
    recall    = round( ms.recall_score( y_true, y_pred ), 3 )
    f1        = round( ms.f1_score( y_true, y_pred ), 3 )

    print(f"Bal Acc: {bal_acc}, Pre: {precision}, Rec: {recall}, F1: {f1}")

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

def get_gradient_boosting_model(X, y, s_scoring):

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
                                        scoring=s_scoring,
                                        n_jobs=-1 )

    random_search.fit(X, y)
    best_params = random_search.best_params_
    model = random_search.best_estimator_

    print(f"Best Params: {best_params}")
    print()

    return model

def predict_women_only( X ):
    y = X["Sex"].apply( lambda k: 1 if k == "female" else 0 )
    return y

def predict_women_and_children( X ):
    y =  X.apply( lambda k: 1 if ( k.Sex == "female" ) or \
                                 ( k.Age <= 13 ) else 0,
                  axis=1 )
    return y

def predict_children_and_rich_women( X ):
    y =  X.apply( lambda k: 1 if (( k.Sex == "female" ) and \
                                    (k.FarePerPerson >= 5) or \
                                 ( k.Age <= 13 )) else 0,
                  axis=1 )
    return y

### MAIN ###

datestamp = "20250129.150949"
df_train = pd.read_csv(f"../data/kaggle/train.clean.{datestamp}.csv")
df_test = pd.read_csv(f"../data/kaggle/test.clean.{datestamp}.csv")

x_colnames = [ "Sex", "Age", "FarePerPerson" ]
y_colname = [ "Survived" ]

X = df_train[ x_colnames ]
y = df_train[ y_colname ].values.ravel()

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.5)
print(f"Data Shapes train: {X_train.shape}, test: {X_test.shape}")

threshold = 0.95

y_test_preds = predict_women_only(X_test)
print_y_scores( y_test, y_test_preds )

#df_test["SurvivedProbability"] = y_proba[:,1]
y_preds = predict_women_only(df_test[x_colnames])
df_test["Survived"] = y_preds
df_test[["PassengerId", "Survived"]].to_csv(f"../data/kaggle/submit.{datestamp}.csv", index=False)
