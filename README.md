# NanoAnalysis
NanoAnalysis is a large scale, ML-driven analysis of inorganic nanoparticles in pre-clinical cancer research.

For more information, please refer to the associated publication: https://www.nature.com/articles/s41565-024-01673-7

If you use this data or code, please kindly cite ` Mendes & Zhang et al. Nature Nanotechnology 19, 867–878 (2024) `

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

Text mining runs in RStudio Version 2022.12.0+353 using functions from tidyverse,  quanteda, tm, ggplot2, ggrepel, and cowplot. All libraries can be installed in Rstudio in the console using
```
install.packages(c("tidyverse","quanteda","quanteda.textmodels","quanteda.textplots","quanteda.textstats","ggplot2","ggrepel","tm","cowplot"))
```

# Descriptions of folders and files
### data
The available data sources include:
* Curated dataset from the literature
* Transformed database ready for machine learning training and analysis
* Dataset of publications from Pubmed and Web of Science general searches on inorganic cancer nanoparticles
* Dataset of publications curated by the research team
* Data on materials and cancer types commonly associated with the inorganic cancer nanoparticle space

### code
* This folder includes core functions that underlie the analysis pipeline, executable examples for users to run, and an interactive notebook to run text mining analysis.
