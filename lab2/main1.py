import pandas as pd

trainData = pd.read_csv("../dataSets/unimel_train.csv") # считываем файл с данными для обучения
testData = pd.read_csv("../dataSets/unimel_test.csv") # считываем файл с данными для тестирования

#print(trainData)
#print(testData)

# данные после считывания будут располагаться в двумерной структуре данных(что-то типа словаря),
# чтобы выделить данные в отдельный вектор нужно просто обратиться к строке по индексу
trainDataGrantStatus = trainData['Grant.Status'][::] # вытаскиваем целевое значение из данных для обучения
testDataGrantStatus = testData['Grant.Status'][::] # вытаскиваем целевое значение из данных для тестирования

print(trainDataGrantStatus)
print(testDataGrantStatus)