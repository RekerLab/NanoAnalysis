### SHAP feature importance analysis
## importing SHAP package, examined model (RF) and other packages
import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier

## load data
df = pd.read_csv('../data/ml_dataset.csv')
X = df.iloc[:, :-3].copy()
y = np.array(df.iloc[:, -2])

## model training
model = RandomForestClassifier(n_estimators=500)
model.fit(X, y)

## building SHAP explainer and extracting SHAP values
explainer = shap.TreeExplainer(model, X)
shap_values = explainer.shap_values(X)
