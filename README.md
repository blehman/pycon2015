Contact:  
  *  Twitter: [@BrianLehman](https://twitter.com/BrianLehman)  
  *  Github: [Blehman](https://github.com/blehman)

# PyCon 2015
Learnings from PyCon 2015.

##Pre conference reading

1.  [Pandas Intro](http://pandas.pydata.org/pandas-docs/version/0.15.2/10min.html)
2.  [Machine Learning in Action](http://www.manning.com/pharrington/)

##Tutorials
###(\#1) Machine Learning with Scikit-Learn (I) w/ Jake VanderPlas
Jake's full presentation using several ipython notebooks is on github: [ML Wisdom I](https://github.com/jakevdp/sklearn_pycon2015). 
#####2015-04-08 lecture notes:  
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
  - Useful: confusion matrix

5. Support vector classifier
  - Goal: draw a line (plane) that splits the data
  - Distance goal: maximize the margin between the points and line 
  - For non linear kernels, we can use 'rbf' (radial basis function),
    which computes a center  

6. Decision Tree and Random Forrest
  - The boundaries (decisions) respond to noise.
  - So overfitting can be a problem if the data contains much noise.
  - *Random Forrest* tries to optimize the answer the boundaries.

7. PCA (Principal component analysis)
  - Useful for dimension reduction
  - Tries to determine the importance of dimensions
  - Question: How much of the variance is preserved? We can select
    dimensions based on how much of the total variance we want to
preserve.

8. K-Means
  - Guess cluster centers
  - Assign points to nearest center
  - Repeat until converged (see his [great demo insie this IPython notebook](https://github.com/jakevdp/sklearn_pycon2015/blob/master/notebooks/04.2-Clustering-KMeans.ipynb))


###(\#2) Machine Learning with Scikit-Learn (II) w/ Olivier Grisel
Olivier's full presentation is available on github: [ML Wisdom II](https://github.com/ogrisel/parallel_ml_tutorial). 

#####2015-04-08 lecture notes: 
0. How to use numpy (basic tutorial).
1. How to deal with heterogenous data.
  - Replace NA w/ median values (see .fillna(median_features) in Random
    notes)  
  - Consideration for factorizing (see example below) categorical variables: if we have
    labels like *British, American, German*, we could represent them as
*(0, 1, 2)*; however, this implicitly assumes that the distance beteen
British and German is larger than the distance between Bristish and
American. Appropriate?  
2. How to massage data when it doesn't fit into a regular numpy array.
3. How to select and evaluate models.
  - ROC Curve for each model is a way to look at the tradeoff between true positive and
    false positives for various tuning. It assumes that the line y=x is
random w/ an area under the curve being 0.5. The area under the ROC
curver > 0.5 suggests the quality of the model. *note: up and left on
the ROC curve is desirable (less false positives and more true
positives)*

  - Cross Validation with a *sufficient* number of folds allows us to test and possibly improve the model. (see %%time below for trade off of increasing the number of folds). The improvement comes from helping us choose, for example, a (regularization) value for C in regression.  

  - GridSearchCV can optimize selected parameters for a model. It uses k folds in cross validation (see GradientBoostingClassifier) to output a mean validation score for each combination of parameters. So the output is a set of scores for each model. Sorting this list based on the on the mean validation score, we can find our *best* combination. (note: setting `n_job=-1` can be help parallelize the process).  

  - Imputer can be used build statistics for each feature, remove the missing values, and then test the affects of data-snooping *note: review this process in the notebook*. 

4. How to classify/cluster text based data.
  - TfidfVectorizer(min_df=2) is set to only keep the documents that has
    words that appear at most twice in the dataset. The output is a
unique sparse matrix that does NOT store the zeros (ie. compressed). We
could use `array.toarray()` or `array.todense()` to bounce between these
representations.

###(\#3) Winning Machine Learning Competitions With Scikit-Learn w/ Ben Hamner
Audience level: Intermediate  
###(\#4) Twitter Network Analysis with NetworkX w/ Sarah Guido, Celia La
Audience level: Intermediate  
###Main Sessions
###Links
Bayesian stat from Allen Downey:  
  * [Think Bayes](http://www.greenteapress.com/thinkbayes/)  
  * [His other books are here](http://www.greenteapress.com/)  

###Random notes
1. Handy ipython tid bits  
  - Transform text to numeric values.
<pre>
  factors, labels = pd.factorize(data.Embarked)
</pre>
  - How to time process 
<pre>
  %%time
</pre>
  - Curl or Read data 
<pre>
  \#!curl -s https://dl.dropboxusercontent.com/u/5743203/data/titanic/titanic_train.csv | head -5
  with open('titanic_train.csv', 'r') as f:
      for i, line in zip(range(5), f):
          print(line.strip())


  \#data = pd.read_csv('https://dl.dropboxusercontent.com/u/5743203/data/titanic/titanic_train.csv')
  data = pd.read_csv('titanic_train.csv')
</pre>
  - Count # of entires per feature
<pre>
  data.count()
</pre>
 - Remove NA, calculate Media, then fill in NA w/ Media
<pre>
  numerical_features = data[['Fare', 'Pclass', 'Age']]

  \# calculate media where .dropna() removes the NA
  median_features = numerical_features.dropna().median()

  \# fill in the na values with the media
  imputed_features = numerical_features.fillna(median_features)
  imputed_features.count()
</pre>

  - To get help with a defined model:
<pre>
  SVC?
</pre>
  - Set the figures to be inline  
<pre>
  %matplotlib inline
</pre>
  - SHIFT + TAB inside a model provides a shortlist of the optional paramaters  
  - SHIFT + ENTER runs a cell and proceeds to next.  
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

- Model validation
  - Split the data into training vs test
  <pre>
    from sklearn.cross_validation import train_test_split
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)

    \#Confusion Matrix:

    neibs = 2
    clf = KNeighborsClassifier(n_neighbors=neibs)
    clf.fit(Xtrain, ytrain)
    ypred = clf.predict(Xtest)
    print(confusion_matrix(ytest, ypred))
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




