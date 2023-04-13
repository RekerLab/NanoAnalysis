# NanoAnalysis
NanoAnalysis is a large scale, ML-driven analysis of inorganic nanoparticles in pre-clinical cancer research

# Dependencies
Abstract put here

Supervised machine learning runs in Python 3.9 using algorithms from [scikit-learn](https://scikit-learn.org/stable/) and [XGBoost](https://xgboost.readthedocs.io/en/stable/). Explainable AI is dependent on [SHAP](https://github.com/slundberg/shap) analysis. [tqdm](https://github.com/tqdm/tqdm) is a useful tool to visually track your job progress. A fresh conda environment can be set up using

```
conda create -n NanoAI python=3.9
conda activate NanoAI
conda install scikit-learn
conda install -c conda-forge py-xgboost
conda install -c conda-forge shap
conda install tqdm
```
Alternatively, users could implement the analysis on cloud-based platforms with pre-configured Python environment, e.g. Googld Colab, required packages can be directly installed using

```
!pip install xgboost
!pip install shap
```

# Descriptions of folders
### data

