import pandas as pd
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
import numpy as np

#trainData = pd.read_csv("../dataSets/unimel_train2.csv") # считываем файл с данными для обучения
testData = pd.read_csv("../dataSets/unimel_test2.csv") # считываем файл с данными для тестирования
#print(testData)
#x = testData[:][0:8]
x = pd.DataFrame(testData,  columns=['Grant.Application.ID', 'Grant.Status',  'Number.of.Successful.Grant.1', 'Number.of.Unsuccessful.Grant.1'])
x = x[::][200:220]
y = testData['Grant.Status'][220]
print(x)
cv = KFold(len(x), n_folds=5, shuffle=True, random_state=241) # задаем генератор разбиений
model = LogisticRegression() # объявляем регрессию
scoring = 'roc_auc' # объявляем метрику качества
results = cross_validation.cross_val_score(model, x, y, cv=cv, scoring=scoring) #делаем кросс-валидацию на основе генерации и регрессии

#print(trainData)

#print(y)