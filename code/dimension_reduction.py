### LDA analysis
## importing LDA function
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

## load data
df = pd.read_csv('../data/ml_dataset.csv')
X = df.iloc[:, :-3].copy()
y_multi = np.array(df.iloc[:, -1])

## building LDA model
lda = LDA(n_components=2).fit(X, y_multi)
lda_space = lda.transform(X)