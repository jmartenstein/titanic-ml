import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.metrics as ms
import statistics as st
import numpy as np

import random
import argparse
import sys

from scipy import interpolate

import sklearn.model_selection as md
import sklearn.neural_network as nn
import sklearn.ensemble as en


### FUNCTIONS ###

def print_crossval_scores( model, X, y, num_folds ):

    kf = md.KFold(n_splits=num_folds)

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

def get_y_scores_string( y_true, y_pred, verbose=False ):

    s_scores = ""

    tn, fp, fn, tp = ms.confusion_matrix(y_true, y_pred).ravel()
    if verbose:
        s_scores = f"Confusion Matrix: tn {tn}, fp {fp}, fn {fn}, tp {tp}\n"

    bal_acc   = round( ms.balanced_accuracy_score( y_true, y_pred ), 3 )
    precision = round( ms.precision_score( y_true, y_pred ), 3 )
    recall    = round( ms.recall_score( y_true, y_pred ), 3 )
    f1        = round( ms.f1_score( y_true, y_pred ), 3 )

    s_scores += f"Bal Acc: {bal_acc}, Pre: {precision}, Rec: {recall}, F1: {f1}"

    return s_scores

def print_feature_importance( model, colnames ):

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_imp_df = pd.DataFrame(
            {'Feature': colnames, 'Gini Importance': importances}).sort_values(
                'Gini Importance', ascending=False)
        print(feature_imp_df)

    return True

def calc_roc_curve( model, X, y, verbose):

    y_preds_proba  = model.predict_proba(X)

    fpr, tpr, thresh = ms.roc_curve(y, y_preds_proba[:,1])
    roc_auc = ms.auc(fpr, tpr)

    # Calculate the geometric mean
    gmeans = np.sqrt(tpr * (1 - fpr))

    # Locate the index of the largest gmean
    index = np.argmax(gmeans)
    threshold = thresh[index]

    tpr_intrp = interpolate.interp1d(thresh, tpr)
    fpr_intrp = interpolate.interp1d(thresh, fpr)

    if verbose:
        print(f"Test AUC {round(roc_auc,3)}")
        print(f"Test optimal threshold {round(threshold,3)} at " \
              f"tpr: {np.round(tpr_intrp(threshold),3)}, " \
              f"fpr: {np.round(fpr_intrp(threshold),3)}")

    return threshold

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

def get_multilayer_perceptron_model(X, y, s_scoring):

    param_dist = {
        'hidden_layer_sizes': [(50,), (100,), (100,100,), (150,), (150,150,), (200,) ],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05, 0.1, 0.5, 1, 10],
        'learning_rate': ['constant','adaptive'],
    }

    mlp_classifier = nn.MLPClassifier(max_iter=2500)
    random_search = md.RandomizedSearchCV( estimator=mlp_classifier,
                                           param_distributions=param_dist,
                                           n_iter=20,
                                           cv=5,
                                           scoring=s_scoring,
                                           n_jobs=-1 )

    random_search.fit(X, y)
    best_params = random_search.best_params_
    best_model = random_search.best_estimator_

    return best_model, best_params

def get_gradient_boost_model(X, y, s_scoring):

    param_dist = {
        'learning_rate': np.arange(0.01, 0.1, 0.01),
        'n_estimators': [50, 75, 100, 125, 150, 175, 200],
        'max_depth': [2, 3, 4, 5, 6],
        'subsample': np.arange(0.2, 0.8, 0.1),
        'max_features': [ 2 ],
        'min_samples_split': [ 30 ]
    }

    gb_classifier = en.GradientBoostingClassifier()
    random_search = md.RandomizedSearchCV( estimator=gb_classifier,
                                        param_distributions=param_dist,
                                        n_iter=50,
                                        cv=5,
                                        scoring=s_scoring,
                                        n_jobs=-1 )

    random_search.fit(X, y)
    best_params = random_search.best_params_
    best_model = random_search.best_estimator_

    return best_model, best_params

def get_random_forest_model( X, y ):
    pass

def validate_model_name( m, learning_models ):

    static_models = [ "women_only",
                      "women_and_children",
                      "children_and_rich_women",
                      "random",
                      "rich_women",
                      "rich_non_misters",
                    ]

    valid_models = static_models + learning_models

    if not m in valid_models :
        print("No valid model. Choose one:")
        print("  " + ", ".join(valid_models))
        sys.exit(1)

def predict_women_only( X ):
    y = X["IsMale"].apply( lambda k: 1 if k == 0 else 0 )
    return y

def predict_women_and_children( X ):
    y =  X.apply( lambda k: 1 if ( k.IsMale == 0 ) or \
                                 ( k.AgeImputed <= 6 ) else 0,
                  axis=1 )
    return y

def predict_rich_women( X ):
    y =  X.apply( lambda k: 1 if (( k.IsMale == 0 ) and \
                                  ( k.Pclass < 3)) else 0,
                  axis=1 )
    return y

def predict_rich_non_misters( X ):
    y =  X.apply( lambda k: 1 if (( k.Mr == 0 ) and \
                                  ( k.Pclass < 3)) else 0,
                  axis=1 )
    return y

def predict_children_and_rich_women( X ):
    y =  X.apply( lambda k: 1 if (( k.IsMale == 0 ) and \
                                  ( k.Pclass < 3) ) or \
                                 ( k.AgeImputed <= 6 ) else 0,
                  axis=1 )
    return y

def predict_random( X ):
    y = [random.randint(0,1) for _ in range(len(X))]
    return y


### MAIN ###

parser = argparse.ArgumentParser( description='Analyze models for Titanic dataset' )
parser.add_argument('-v', '--verbose', action='store_true', help='Add verbose output')
parser.add_argument('-m', '--model', help='Name of model to use')
parser.add_argument('-n', '--number', help='Times to split / analyze data with model' )
parser.add_argument('-a', '--all_features', action='store_true', help='User all features in model')
parser.set_defaults( verbose=False, model='women_only', number=3 )

args = vars( parser.parse_args() )
#print(args)

learning_models = [ 'gradient_boosted',
                    'multilayer_perceptron'
                  ]

validate_model_name( args["model"], learning_models )
s_function = "predict_" + args["model"]

datestamp = "20250308.163757"
df_train = pd.read_csv(f"../data/kaggle/train.clean.{datestamp}.csv")
df_test = pd.read_csv(f"../data/kaggle/test.clean.{datestamp}.csv")

if args["all_features"]:

    columns_to_drop = [ "Survived", "Died", "Name", "Title", "TitleGrouped", "LastName", "Cabin",
                        "Ticket", "Sex", "Age", "Embarked", "PassengerId", "CabinDeck",
                        "CabinRoom", "TicketConfirmedSurvived", "LastNameConfirmedSurvived",
                        "LastNameConfirmedDied", "TicketConfirmedDied" ]
    #columns_to_drop += [ "AgeImputed", "FarePerPerson", "FppMinMax", "SibSp", "Parch", "SexOrd",
    #                     "AgeMinMax", "TicketConfirmedSurvived", "GroupSurvived",
    #                     "TicketConfirmedDied", "GroupDied", "Died" ]
    x_colnames = [i for i in df_train.columns if i not in columns_to_drop]

else:
    x_colnames = [ "IsMale", "Fare", "TitleOrd", "TicketSurvivorScore", "P3orDeadTitle" ]

y_colname = [ "Survived" ]

X = df_train[ x_colnames ]
y = df_train[ y_colname ].values.ravel()

X_train, X_test, y_train, y_test = md.train_test_split( X, y, test_size=0.2)
if args['verbose']:
    print(f"Data Shapes train: {X_train.shape}, test: {X_test.shape}")


if args["model"] in learning_models:

    if s_function == "predict_gradient_boosted":
        model, params = get_gradient_boost_model( X_train, y_train, "f1_weighted" )

    if s_function == 'predict_multilayer_perceptron':
        model, params = get_multilayer_perceptron_model( X_train, y_train, "f1_weighted" )

    if args['verbose']:
        print(f"Best Params: {params}")
        print_feature_importance( model, x_colnames )

    threshold = calc_roc_curve( model, X_test, y_test, args["verbose"] )
    #threshold = 0.65

    y_test_preds_proba = model.predict_proba(X_test)
    y_test_preds = (y_test_preds_proba[:,1] > threshold).astype(int)

    s_scores = get_y_scores_string( y_test, y_test_preds, args['verbose'] )

    y_preds_proba = model.predict_proba(df_test[x_colnames])
    y_preds = (y_preds_proba[:,1] > threshold).astype(int)

else:

    y_test_preds = locals()[s_function](X_test)
    s_scores = get_y_scores_string( y_test, y_test_preds, args['verbose'] )
    y_preds = locals()[s_function](df_test[x_colnames])

print(f"Model: {args['model']}, {s_scores}")

df_test["Survived"] = y_preds
df_test[["PassengerId", "Survived"]].to_csv(
    f"../data/kaggle/submit.{args['model']}.csv",
    index=False)
