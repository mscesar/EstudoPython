import pandas as pd

base = pd.read_csv('census.csv')

base.info()

base.head(5)

base.describe()

base.columns.values

variaveis = base.iloc[:, 0:14].values

resultado = base.iloc[:, 14].values
                
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_variaveis = LabelEncoder()
variaveis[:, 1] = labelencoder_variaveis.fit_transform(variaveis[:, 1])
variaveis[:, 3] = labelencoder_variaveis.fit_transform(variaveis[:, 3])
variaveis[:, 5] = labelencoder_variaveis.fit_transform(variaveis[:, 5])
variaveis[:, 6] = labelencoder_variaveis.fit_transform(variaveis[:, 6])
variaveis[:, 7] = labelencoder_variaveis.fit_transform(variaveis[:, 7])
variaveis[:, 8] = labelencoder_variaveis.fit_transform(variaveis[:, 8])
variaveis[:, 9] = labelencoder_variaveis.fit_transform(variaveis[:, 9])
variaveis[:, 13] = labelencoder_variaveis.fit_transform(variaveis[:, 13])

onehotencoder = OneHotEncoder(categorical_features = [1,3,5,6,7,8,9,13])
variaveis = onehotencoder.fit_transform(variaveis).toarray()

labelencoder_resultado = LabelEncoder()
resultado = labelencoder_resultado.fit_transform(resultado)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
variaveis = scaler.fit_transform(variaveis)

from sklearn.model_selection import train_test_split
variaveis_treinamento, variaveis_teste, resultado_treinamento, resultado_teste = train_test_split(variaveis, resultado, test_size=0.3, random_state=0)

#NAIVE BAYES

from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(variaveis_treinamento, resultado_treinamento)
previsoes = classificador.predict(variaveis_teste)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(resultado_teste, previsoes)
matriz = confusion_matrix(resultado_teste, previsoes)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(resultado_teste,previsoes))  
print(classification_report(resultado_teste,previsoes))  
print(accuracy_score(resultado_teste, previsoes)) 

#RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier
classificador = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
classificador.fit(variaveis_treinamento, resultado_treinamento)
previsoes = classificador.predict(variaveis_teste)

# Model Evaluation Metrics 
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
print('Accuracy Score : ' + str(accuracy_score(resultado_teste, previsoes)))
print('Precision Score : ' + str(precision_score(resultado_teste, previsoes)))
print('Recall Score : ' + str(recall_score(resultado_teste, previsoes)))
print('F1 Score : ' + str(f1_score(resultado_teste, previsoes)))

from sklearn.metrics import classification_report  
print(classification_report(resultado_teste,previsoes))  

#Confusion Matrix
from sklearn.metrics import confusion_matrix
print('Confusion Matrix : \n' + str(confusion_matrix(resultado_teste, previsoes)))

confusion = confusion_matrix(resultado_teste, previsoes)

#Cross Validation
from sklearn.model_selection import cross_val_score  
all_accuracies = cross_val_score(estimator = classificador, X = variaveis_treinamento, y = resultado_treinamento, cv = 10)
print(all_accuracies)
print(all_accuracies.mean())

from sklearn.model_selection import cross_val_score
rf = RandomForestClassifier(n_estimators = 100)
scores = cross_val_score(rf, X_train, Y_train, cv = 10, scoring = "accuracy")

print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())

metricas = ['accuracy', 'average_precision', 'f1', 'precision', 'recall', 'roc_auc']

from sklearn.model_selection import cross_validate
cross_validate(classificador, variaveis_treinamento, resultado_treinamento, return_train_score = False, scoring = metricas)

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
predictions = cross_val_predict(random_forest, X_train, Y_train, cv=3)
confusion_matrix(Y_train, predictions)

#Random Search With Cross Validation
import numpy as np
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 100, num = 10)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(train_features, train_labels)

rf_random.best_params_

#Grid Search With Cross Validation
classificador = RandomForestClassifier()

from sklearn.model_selection import GridSearchCV

classificador.get_params().keys()

grid_param = {  
    'n_estimators': [100, 300, 500, 800, 1000],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False]
}

grid_param = {"max_depth": range(1,25),
              "criterion": ["gini", "entropy"],
              "max_leaf_nodes": [None, 2, 3, 5, 7, 9, 12, 15]
             }

grid_param = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}

grid_param = {
            "criterion": ['entropy', 'gini'],
            "n_estimators": [25, 50, 75],
            "bootstrap": [False, True],
            "max_depth": [3, 5, 10],
            "max_features": ['auto', 0.1, 0.2, 0.3]
}

gd_sr = GridSearchCV(estimator = classificador,  
                     param_grid = grid_param,
                     scoring = 'accuracy',
                     cv = 10,
                     return_train_score = True,
                     n_jobs = -1)



gd_sr.fit(variaveis_treinamento, resultado_treinamento) 

best_parameters = gd_sr.best_params_  
print(best_parameters)

best_result = gd_sr.best_score_  
print(best_result)

pd.set_option('max_columns',200)
gd_sr_results = pd.DataFrame(gd_sr.cv_results_)

#SVM

from sklearn.svm import SVC
classificador = SVC(kernel = 'linear', random_state = 1)
classificador.fit(variaveis_treinamento, resultado_treinamento)
previsoes = classificador.predict(variaveis_teste)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(resultado_teste, previsoes)
matriz = confusion_matrix(resultado_teste, previsoes)

#REDES NEURAIS

from sklearn.neural_network import MLPClassifier
classificador = MLPClassifier(verbose = True, max_iter=1000, tol=0.000010)
classificador.fit(variaveis_treinamento, resultado_treinamento)
previsoes = classificador.predict(variaveis_teste)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(resultado_teste, previsoes)
matriz = confusion_matrix(resultado_teste, previsoes)

#REGRESSÃO LOGISTICA

#Grid Search
from sklearn.model_selection import GridSearchCV
clf = LogisticRegression()
grid_values = {'penalty': ['l1', 'l2'],'C':[0.001,.009,0.01,.09,1,5,10,25]}
grid_clf_acc = GridSearchCV(clf, param_grid = grid_values,scoring = 'recall')
grid_clf_acc.fit(X_train, y_train)

#Predict values based on new parameters
y_pred_acc = grid_clf_acc.predict(X_test)

# New Model Evaluation Metrics 
print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred_acc)))
print('Precision Score : ' + str(precision_score(y_test,y_pred_acc)))
print('Recall Score : ' + str(recall_score(y_test,y_pred_acc)))
print('F1 Score : ' + str(f1_score(y_test,y_pred_acc)))

#Logistic Regression (Grid Search) Confusion matrix
confusion_matrix(y_test,y_pred_acc)