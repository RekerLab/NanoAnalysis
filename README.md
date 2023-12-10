# NanoAnalysis
NanoAnalysis is a large scale, ML-driven analysis of inorganic nanoparticles in pre-clinical cancer research.

# Dependencies
Supervised machine learning runs in Python 3.9 using algorithms from [scikit-learn](https://scikit-learn.org/stable/) and [XGBoost](https://xgboost.readthedocs.io/en/stable/). Explainable AI is dependent on [SHAP](https://github.com/slundberg/shap) analysis. [tqdm](https://github.com/tqdm/tqdm) is a useful tool to visually track your job progress. A fresh conda environment can be set up using

```
conda create -n NanoAna python=3.9 pandas
conda activate NanoAna
conda install scikit-learn
conda install -c conda-forge py-xgboost
conda install -c conda-forge shap
conda install tqdm
```
Alternatively, users could implement the analysis on cloud-based platforms with pre-configured Python environment, e.g. Google Colab, and required packages can be installed using

```
!pip install xgboost
!pip install shap
```

# Descriptions of folders
### data
* The available data sources include a curated dataset from the literature, as well as a transformed database that is ready for machine learning training and analysis.

### code
* This folder includes core functions that underlie the analysis pipeline, along with executable examples for users to run.
