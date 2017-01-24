import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#Loading Dataset done

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = [ 'sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

#summarizing Dataset

print(dataset.shape)
print(dataset.head(20))

#statistical summary
print(dataset.describe())
print(dataset.groupby('class').size())

#Data Visualization
#box & whisker plot
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

dataset.hist()
plt.show()

#scatter plot matrix
scatter_matrix(dataset)
plt.show()

data_arr = dataset.values;
X = data_arr[:,0:4]
Y = data_arr[:,4]
validation_size = 0.20
seed = 7
X_Train, X_Validate, Y_Train, Y_Validate = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

seed = 7
scoring='accuracy'


#building models
models = []
models.append(('LR',LogisticRegression()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('SVM',SVC()))
models.append(('NB',GaussianNB()))

#evaluate each model
result = []
names = []

for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state = seed)
	cv_results = model_selection.cross_val_score(model, X_Train, Y_Train, cv=kfold, scoring=scoring)
	result.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name,cv_results.mean(),cv_results.std())
	print(msg)
	



















