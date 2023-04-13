# NanoAI
NanoAI is a large scale analysis of inorganic nanoparticles in pre-clinical cancer research

# Dependencies
Abstract

## Supervised Machine Learning
Runs in Python 3.9 using algorithms from [scikit-learn](https://scikit-learn.org/stable/) and [XGBoost](https://xgboost.readthedocs.io/en/stable/). [tqdm](https://github.com/tqdm/tqdm) is a useful tool to visually track your job progress. A fresh conda environment can be set up using

```
conda create -n NanoAI python=3.9
conda activate NanoAI
conda install scikit-learn
conda install -c conda-forge py-xgboost
conda install tqdm
```

