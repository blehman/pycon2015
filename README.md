# PyCon 2015
Learnings from PyCon 2015.

##Pre conference reading

1.  [Pandas Intro](http://pandas.pydata.org/pandas-docs/version/0.15.2/10min.html)
2.  [Machine Learning in Action](http://www.manning.com/pharrington/)

##Tutorials
###Machine Learning with Scikit-Learn (I) w/ Jake VanderPlas
These notes are taken from Jake's April 8th, 2015 presentation at PyCon.
Below are just a few notes on pieces of the talk. Jake's full presentation using several ipython notebooks is on github: [ML wisdom](https://github.com/jakevdp/sklearn_pycon2015).

####2015-04-08 notes:  
1. Three major steps emerge:  
  *  instantiate the model
  *  fit the model
  *  use the model to predict    

2. Interesting method:  
  *  kNN has a predit_proba method!

3. Supervised vs Unspervised
Unsupervised Learning addresses a different sort of problem. Here the data has no labels, and we are interested in finding similarities between the objects in question. You can think of unsupervised learning as a means of discovering labels from the data itself.
  - define model and instantiate class
  - fit the model (no lables)
  - use the model to predict

4. Model validation
  - Split the data into training vs test
<pre>
from sklearn.cross_validation import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
</pre>
  - Confusion Matrix:
<pre>
neibs = 2
clf = KNeighborsClassifier(n_neighbors=neibs)
clf.fit(Xtrain, ytrain)
ypred = clf.predict(Xtest)
print(confusion_matrix(ytest, ypred))
</pre>

5. Support vector classifier
  - Goal: draw a line (plane) that splits the data
  - Distance goal: maximize the margin between the points and line 
  - For non linear kernels, we can use 'rbf' (radial basis function),
    which computes a center  

6. Decision Tree 
  - The boundaries (decisions) respond to noise.
  - So overfitting can be a problem if the data contains much noise.
  - *Random Forrest* tries to optimize the answer the boundaries.

7. Handy ipython tid bits  
  - To get help with a defined model:
<pre>
SVC?
</pre>
  - Set the figures to be inline  
<pre>
%matplotlib inline
</pre>
  - SHIFT + TAB inside a model provides a shortlist of the optional paramaters  
  - grab the iris dataset
<pre> 
from sklearn.datasets import load_iris 
iris = load_iris()
</pre>
- start to consider numpy arrays and features from the dataset  
<pre>
import numpy as np  
iris.keys()
iris.data.shape
print iris.data[0:3,0]  
print iris.data  
</pre>
- use models from scikit-learn
Notice that we input data from the model y = 2x + 1 and this model is
accurately predicted.
<pre>
from sklearn.linear_model import LinearRegression
model = LinearRegression(normalize=True)
x = np.arange(10)
X = x[:, np.newaxis]
y = 2 * x + 1
model.fit(X, y)
print(model.coef_)
print(model.intercept_)
</pre>

- kNN (very interesting addition here: probabilistic predictions on the
  last line)
<pre>
  from sklearn import neighbors, datasets

  iris = datasets.load_iris()
  X, y = iris.data, iris.target

  \#instantiate the model
  knn = neighbors.KNeighborsClassifier(n_neighbors=5)

  \# fit the model
  knn.fit(X, y)

  \# use the model to predict
  knn.predit([[3, 5, 4, 2],])


  \# What kind of iris has 3cm x 5cm sepal and 4cm x 2cm petal?
  \# call the "predict" method:
  result = knn.predict([[3, 5, 4, 2],])

  print(iris.target_names[result])
  print iris.target_names
  print knn.predict_proba([[3, 5, 4, 2],])
</pre>

- Random Forrest
<pre>
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=300, centers=4,
                  random_state=0, cluster_std=1.0)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='rainbow');
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=0)
visualize_tree(clf, X, y, boundaries=False);
</pre>

###Machine Learning with Scikit-Learn (II) w/ Olivier Grisel
Audience level: Intermediate  
###Winning Machine Learning Competitions With Scikit-Learn w/ Ben Hamner
Audience level: Intermediate  
###Twitter Network Analysis with NetworkX w/ Sarah Guido, Celia La
Audience level: Intermediate  



