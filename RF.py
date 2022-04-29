# -*- coding: utf-8 -*-

import os
import sys
import csv
import pandas as pd
import geopandas as gpd
import pickle
import numpy as np
np.random.seed(123)
import scipy
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.inspection import permutation_importance
import seaborn as sns




    
def ParameterTuning(X,y,k):
    tuned_parameters = {
    'max_features': [2, 3, 4, 5],
    'n_estimators': [300,400,500],

    }
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=24)
    rfc=RandomForestClassifier(random_state=42)
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=tuned_parameters, cv= k,n_jobs=-1)
    CV_rfc.fit(X_train, y_train)

    print("tuned hpyerparameters :(best parameters) ",CV_rfc.best_params_)
    print("accuracy :",CV_rfc.best_score_)
    
   


if __name__=="__main__":
    
    
    data = pickle.load(open('Stockholm_data_for_training.pkl','rb'))

    statistics = data.describe()
    y = data['TtB_class']
    columns = ['TtB_class']
    # columns = ['TtB_class','TtB_class1','Area', 'FID']
    X = data.drop(columns,axis=1)
    n = X.shape[1]
    X = np.array(X)
    
    
    ###parameter tuning
    # ParameterTuning(X,y,5)  ####optimal parameter: number of trees(500),max features(2)
    

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3,random_state=24) # 70% training and 30% test    
    clf = RandomForestClassifier(n_estimators=500,random_state=0, max_features=2)
    
    model = clf.fit(X_train,Y_train)
    ypredict = clf.predict(X_test)
    #    #evaluate
    acc = accuracy_score(Y_test,ypredict)
    cnf_matrix = confusion_matrix(Y_test,ypredict)
    f1 = f1_score(Y_test, ypredict, average='binary')
    precision = precision_score(Y_test, ypredict, average='binary')
    recall = recall_score(Y_test, ypredict, average='binary')
#    
    print("X1: accuracy using test = {:.2f}%".format(acc*100))
    print(cnf_matrix)
    print(clf.feature_importances_)
    print('F1:')
    print(f1)
    print('precision:')
    print(precision)
    print('recall:')
    print(recall)
    
    result = permutation_importance(model, X_test, Y_test, n_repeats=10, random_state=0, scoring='accuracy')
    feature_names = ['POI_density', 'Stop_density', 'End_battery', 'Hour_day', 'Day_week',
        'Distance_motorway', 'Distance_pedestrian', 'Distance_residential', 'Distance_POI', 'Distance_stop']
    perm = np.transpose(result.importances)
    df = pd.DataFrame(perm,columns=feature_names)
    plt.figure(figsize=(10,8))
    # plt.barh(feature_names, result.importances_mean)

    sns.boxplot(data=df, orient='h')
    plt.rcParams["font.family"] = "Times New Roman"
    plt.tick_params(labelsize=18)
    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 20,
    }
    # plt.savefig('perm_RF_accuracy.png', dpi=1000, bbox_inches='tight')

    print("OK")    
