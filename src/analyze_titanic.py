import sklearn.metrics as ms
import matplotlib.pyplot as plt
import pandas as pd
import statistics as st
import seaborn as sns

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

    return True

def plot_roc_curve( model, X, y ):

    probs = model.predict_proba(X)
    preds = probs[:,1]

    fpr, tpr, threshold = ms.roc_curve(y, preds)
    roc_auc = ms.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')

    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1],'r--')

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    plt.legend(loc = 'lower right')
    plt.show()

    return True


### MAIN ###

df = pd.read_csv("../data/kaggle/train_expanded.csv")
print(f"training data shape: {df.shape}")

x_colnames = [ "Pclass", "Title", "Sex" ]
y_colname = [ "Survived" ]

X_ = df[ x_colnames ]
y  = df[ y_colname ].values.ravel()

enc = preprocessing.OrdinalEncoder(max_categories=5)
X = enc.fit_transform(X_)

#corr_matrix = pd.DataFrame(X, columns=x_colnames).corr()
#sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu')
#plt.show()


X_train, X_test, y_train, y_test = train_test_split( X, y, \
                                     test_size=0.3, random_state=42)

#classifier = svm.SVC( kernel='rbf', C=1, probability=True )
classifier = ensemble.GradientBoostingClassifier(max_leaf_nodes=50, max_depth=3)
#classifier = ensemble.RandomForestClassifier(n_estimators=100)

model = classifier.fit(X_train, y_train)

print_accuracy_scores(model, X_test, y_test, 5)
print_feature_importance(model, x_colnames)
plot_roc_curve( model, X, y )

