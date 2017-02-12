from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
import pandas as pd
url = "https://goo.gl/eGVE1I"

array = pd.read_csv(url).values
X = array[:,0:8]
Y = array[:,8]
results = cross_validation.cross_val_score(LogisticRegression(), X, Y, cv=KFold(len(X), n_folds=5, shuffle=True, random_state=241), scoring='roc_auc')
for i, C in enumerate((0.001, 0.01, 0.1, 1.0, 10.0, 100.0)):
        clf_l1_LR = LogisticRegression(C=C, penalty='l1')
        clf_l1_LR.fit(X, Y)
        coef_l1_LR = clf_l1_LR.coef_
        print("C=%.3f" % C)
        print("score with L1 penalty: %.4f" % clf_l1_LR.score(X, Y))
clf_l1_LR = LogisticRegression(C=0.01, penalty='l1')
results1 = cross_validation.cross_val_score(clf_l1_LR, X, Y, cv=KFold(len(X), n_folds=5, shuffle=True, random_state=241), scoring='roc_auc')