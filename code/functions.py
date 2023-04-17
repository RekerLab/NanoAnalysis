# -*- coding: utf-8 -*-
### Loading Libraries
import pandas as pd
import numpy as np
from tqdm import tqdm

### supervised machine learning models and evaluation
## importing machine learning libraries and tools
import xgboost
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, matthews_corrcoef

## defining utility functions
def proba_output(model, X_test):
  if isinstance(model, LinearSVC):
    proba = list(model.decision_function(X_test))
  else:
    proba = list(1 - model.predict_proba(X_test)[:, 0])
  return proba

def data_stand(X_train, X_test):
  scaler = StandardScaler()
  scaler.fit(X_train)
  X_train_stand = scaler.transform(X_train)
  X_test_stand = scaler.transform(X_test)
  return X_train_stand, X_test_stand

def model_eval(model, X_train, y_train, X_test, y_test):
  model.fit(X_train, y_train)
  proba = proba_output(model, X_test)
  pred = list(model.predict(X_test))
  y_true = list(y_test)
  return [proba, pred, y_true]

def result_update(proba_list, pred_list, y_true_list, proba_score, pred_score, y_true_score, n, r):
  if n == 0:
    proba_list.append(proba_score)
    pred_list.append(pred_score)
    y_true_list.append(y_true_score)
  else:
    proba_list[r].extend(proba_score)
    pred_list[r].extend(pred_score)
    y_true_list[r].extend(y_true_score)
  return proba_list, pred_list, y_true_list

## repeated cross validation
def repeated_clf_cv(X, y, models, repeats=10, stand=True):
  probas = {name: [] for name in [type(model).__name__ for model in models]}
  preds = {name: [] for name in [type(model).__name__ for model in models]}
  y_trues = {name: [] for name in [type(model).__name__ for model in models]}
  r = 0
  y = np.array(y)
  for _ in tqdm(range(repeats), desc='repeated cv'):
    kf = KFold(n_splits=10, shuffle=True)
    n = 0
    for train_index, test_index in kf.split(X):
      X_train, X_test = X.iloc[train_index], X.iloc[test_index]
      y_train, y_test = y[train_index], y[test_index]
      if stand:
        X_train, X_test = data_stand(X_train, X_test)
      for model in models:
        results = model_eval(model, X_train, y_train, X_test, y_test)
        model_name = type(model).__name__
        probas[model_name], preds[model_name], y_trues[model_name] = result_update(probas[model_name], preds[model_name], y_trues[model_name],
                                                                                   *results, n, r)
      n += 1
    r += 1
  return [probas, preds, y_trues]

## label shuffling
def label_shuffling_eval(X, y, repeats=10, stand=True, model=RandomForestClassifier(n_estimators=500)):
  proba_ctrl, pred_ctrl, y_true_ctrl = [], [], []
  proba_y_shuf, pred_y_shuf, y_true_y_shuf = [], [], []
  r = 0
  y = np.array(y)
  for _ in tqdm(range(repeats), desc='label shuffling'):
    kf = KFold(n_splits=10, shuffle=True)
    n = 0
    for train_index, test_index in kf.split(X):
      X_train, X_test = X.iloc[train_index], X.iloc[test_index]
      y_train, y_test = y[train_index], y[test_index]
      if stand:
        X_train, X_test = data_stand(X_train, X_test)
      
      # ctrl
      results = model_eval(model, X_train, y_train, X_test, y_test)
      proba_ctrl, pred_ctrl, y_true_ctrl = result_update(proba_ctrl, pred_ctrl, y_true_ctrl, 
                                                         *results, n, r)
      # y shuffling
      y_train_shuf = np.random.permutation(y_train)
      results = model_eval(model, X_train, y_train_shuf, X_test, y_test)
      proba_y_shuf, pred_y_shuf, y_true_y_shuf = result_update(proba_y_shuf, pred_y_shuf, y_true_y_shuf,
                                                               *results, n, r)
      n += 1
    r += 1
  return [[proba_ctrl, pred_ctrl, y_true_ctrl],
          [proba_y_shuf, pred_y_shuf, y_true_y_shuf]]

## feature ablation challenge
# feature ablation approach
def mode_func(model, mode, X, y):
  y = np.array(y)
  model.fit(X, y)
  if mode == 'random':
    func = np.random.randint(len(X[0]))
  elif mode == 'import':
    func = np.argmax(model.feature_importances_)
  X = np.delete(X, func, axis=1)
  return X, list(np.argmax(model.oob_decision_function_, axis=1))

# core function
def feature_ablation_eval(X, y, repeats=10, stand=True, model=RandomForestClassifier(n_estimators=500, oob_score=True)):
  y = np.array(y)
  oob_decision = {'random': [[] for i in range(repeats)], 
                  'important': [[] for i in range(repeats)]}
  for r in tqdm(range(repeats), desc='feature_ablation'): #, r, repeats
    X_reset = np.array(X) # array and dataframe have different shape reading algorithms
    X_random, X_import = X_reset.copy(), X_reset.copy()
    for _ in range(len(X_reset[0])): # iterate all features
      # remove features in a random manner
      X_random, decision_func = mode_func(model, 'random', X_random, y)
      oob_decision['random'][r].append(decision_func)

      # remove features in a feature-importance based manner
      X_import, decision_func = mode_func(model, 'import', X_import, y)
      oob_decision['important'][r].append(decision_func)
  return oob_decision

# result analysis
def ablation_results(mode, metric_func, metric_name, results, y_true):
  repeats = len(results[mode])
  raw_score = np.array(
      [[metric_func(y_true, results[mode][r][i]) for r in range(repeats)] for i in range(len(results[mode][0]))]
  )
  score_mean = [np.mean(raw_score[i, :]) for i in range(len(results[mode][0]))]
  score_std = [np.std(raw_score[i, :]) for i in range(len(results[mode][0]))]
  output_dic = {metric_name: [score_mean, score_std]}
  return output_dic
