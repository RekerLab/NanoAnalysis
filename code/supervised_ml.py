from functions import *

## load data
df = pd.read_csv('../data/ml_dataset.csv')
X = df.iloc[:, :-3].copy()
y = np.array(df.iloc[:, -2])

## model survey
# define classifiers and exemplary parameter searching space
model_dict = {
    'NB': GaussianNB(),
    'kNN': KNeighborsClassifier(),
    'SVC': SVC(probability=True),
    'DT': DecisionTreeClassifier(),
    'RF': RandomForestClassifier(),
    'XGBoost': XGBClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'MLP': MLPClassifier()
}

param_dict = {
    'NB': {},
    'kNN': {'model__n_neighbors': [3, 4, 5]},
    'SVC': {'model__kernel': ['linear', 'poly', 'rbf', 'sigmoid']},
    'DT': {'model__criterion': ['gini', 'entropy', 'log_loss']},
    'RF': {'model__n_estimators': [100, 500, 1000]},
    'XGBoost': {'model__n_estimators': [100, 500, 1000]},
    'AdaBoost': {'model__n_estimators': [50, 100, 500]},
    'MLP': {
        'model__hidden_layer_sizes': [(100,), (100, 100), (50, 50, 50), (50, 100, 50), (100, 100, 100)],
        'model__activation': ['identity', 'logistic', 'tanh', 'relu'],
        'model__solver': ['lbfgs', 'sgd', 'adam'],
        'model__alpha': [0.0001, 0.0005, 0.001],
        'model__max_iter': [200, 500, 1000]}
}

# evaluate model performance
hyper_opt(model_dict, param_dict, X, y, repeats=5, n_splits=5)

# label shuffling
repeats = 10
results = label_shuffling_eval(X, y, repeats=repeats, stand=True)
entries = []
mode = ['ctrl', 'y_shuffle']
for i in range(repeats):
  for m in range(len(mode)):
    result = results[m]
    # ctrl
    roc_score = roc_auc_score(result[2][i], result[0][i])
    bac_score = balanced_accuracy_score(result[2][i], result[1][i])
    f1 = f1_score(result[2][i], result[1][i])
    mcc_score = matthews_corrcoef(result[2][i], result[1][i])
    pr_score = average_precision_score(result[2][i], result[0][i])
    cohen_score = cohen_kappa_score(result[2][i], result[1][i])    
    entries.append([mode[m], i, roc_score, bac_score, f1, mcc_score, pr_score, cohen_score])
label_shuffling_df = pd.DataFrame(data=entries, columns=['mode', 'repeats', 
                                                         'roc_auc_score', 'balanced_accuracy_score', 'f1_score', 'mcc_score',
                                                         'pr_auc_score', 'cohen_kappa'])

# feature ablation
repeats = 10
results = feature_ablation_eval(X, y, repeats=repeats, stand=True)
metric_func = [balanced_accuracy_score, f1_score, matthews_corrcoef, cohen_kappa_score]
metric_name = ['balanced_accuracy_score', 'f1_score', 'mcc_score', 'cohen_kappa']
entries = []
for mode in ['random', 'important']:
  for i in range(len(metric_func)):
    metric_sum = ablation_results(mode, metric_func[i], metric_name[i], results, y)
    for key, value in metric_sum.items():
      metric, stats = key, value
    entries.append([mode, metric, *stats])
feature_abaltion_df = pd.DataFrame(data=entries, columns=['mode', 'metric', 'avg', 'std'])









