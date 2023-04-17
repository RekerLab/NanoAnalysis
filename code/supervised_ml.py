from functions import *

## load data
df = pd.read_csv('../data/ml_dataset.csv')
X = df.iloc[:, :-3].copy()
y = np.array(df.iloc[:, -2])

# repeated 10-fold cross validation
models = [XGBClassifier(), GaussianNB(), KNeighborsClassifier(n_neighbors=3), LinearSVC(),
          DecisionTreeClassifier(), RandomForestClassifier(n_estimators=500), AdaBoostClassifier(), MLPClassifier()]
# evaluation
model_names = [type(model).__name__ for model in models]
repeats = 10
probas, preds, y_trues = repeated_clf_cv(X, y, models, repeats=repeats, stand=False)
entries = []
for name in model_names:
  for i in range(repeats):
    roc_score = roc_auc_score(y_trues[name][i], probas[name][i])
    bac_score = balanced_accuracy_score(y_trues[name][i], preds[name][i])
    f1 = f1_score(y_trues[name][i], preds[name][i])
    mcc_score = matthews_corrcoef(y_trues[name][i], preds[name][i])
    entries.append([name, i, roc_score, bac_score, f1, mcc_score])
repeat_cv_df = pd.DataFrame(data=entries, columns=['model', 'repeats', 'roc_auc_score', 'balanced_accuracy_score', 'f1_score', 'mcc_score'])

# label shuffling
repeats = 10
results = label_shuffling_eval(X, y, repeats=repeats, stand=False)
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
    entries.append([mode[m], i, roc_score, bac_score, f1, mcc_score])
label_shuffling_df = pd.DataFrame(data=entries, columns=['mode', 'repeats', 'roc_auc_score', 'balanced_accuracy_score', 'f1_score', 'mcc_score'])

# feature ablation
repeats = 10
results = feature_ablation_eval(X, y, repeats=repeats, stand=False)
metric_func = [balanced_accuracy_score, f1_score, matthews_corrcoef]
metric_name = ['balanced_accuracy_score', 'f1_score', 'mcc_score']
entries = []
for mode in ['random', 'important']:
  for i in range(len(metric_func)):
    metric_sum = ablation_results(mode, metric_func[i], metric_name[i], results, y)
    for key, value in metric_sum.items():
      metric, stats = key, value
    entries.append([mode, metric, *stats])
feature_abaltion_df = pd.DataFrame(data=entries, columns=['mode', 'metric', 'avg', 'std'])