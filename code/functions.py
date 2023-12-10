# -*- coding: utf-8 -*-
### Loading Libraries
import pandas as pd
import numpy as np
from tqdm import tqdm

### supervised machine learning models and evaluation
## importing machine learning libraries and tools
import xgboost
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, matthews_corrcoef, average_precision_score, cohen_kappa_score

## defining utility functions
def proba_output(model, X_test):
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

def init_dicts(count_variable, model_name, count_empty_list):
  return [dict(zip(model_name, [[[] for x in range(count_empty_list)] for y in range(len(model_name))])) for z in range(count_variable)]

def flat_list(list1, list2):
  return [item for sublist in zip(list1, list2) for item in sublist]

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

## hyperparameter optimization
def hyper_opt(model_dict, param_dict, X, y, repeats=5, n_splits=5):
  model_list = list(model_dict.keys())
  traj = dict(zip(model_list, [[] for i in range(len(model_list))]))
  proba, true_label = init_dicts(2, model_list, repeats)
  #output, result_dict, labels_dict = init_dicts(4, model_list, repeats)
  for _ in tqdm(range(repeats), desc='repeat'):
    proba_repeat =[]
    true_repeat = []
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    for train_index, valid_index in tqdm(kf.split(X, y), desc='{}-fold'.format(n_splits)): # split the whole dataset into training (developing) set and external validation set
      X_train, y_train = X.iloc[train_index], y[train_index]
      X_valid, y_valid = X.iloc[valid_index], y[valid_index]
      # define grid search settings
      for model_name in tqdm(model_list, desc='model optimization'):
        model = model_dict[model_name]
        param = param_dict[model_name]
        scaler = StandardScaler()
        pipe = Pipeline([('scaler', scaler), ('model', model)])
        grid = GridSearchCV(pipe, param, cv=5, scoring='roc_auc') # 5-fold internal stratified cross validation for hyperparameter optimization
        grid.fit(X_train, y_train)
        traj[model_name].append(pd.DataFrame(grid.cv_results_)) # keep track of the best performing estimator
        raw_proba = [1-i for i in grid.predict_proba(X_valid)[:, 0]] # save predicted probability on the external validation set
        proba[model_name][_].extend(raw_proba)
        true_label[model_name][_].extend(y_valid)
  # export results
  for model_name in tqdm(model_list, desc='model output generation'):
    df = pd.DataFrame(flat_list(proba[model_name], true_label[model_name]))
    df = df.transpose()
    df.columns = flat_list(['Proba_{}'.format(i) for i in range(repeats)], ['Label_{}'.format(i) for i in range(repeats)])
    df.to_csv('{}_output.csv'.format(model_name), index=False)
    traj_df = pd.concat(traj[model_name])
    traj_df.to_csv('{}_traj.csv'.format(model_name), index=False)
  

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
