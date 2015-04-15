Contact:  
  *  Twitter: [@BrianLehman](https://twitter.com/BrianLehman)  
  *  Github: [blehman](https://github.com/blehman)

# PyCon 2015
Learnings from PyCon 2015.

##Pre conference reading

1.  [Pandas Intro](http://pandas.pydata.org/pandas-docs/version/0.15.2/10min.html)
2.  [Machine Learning in Action](http://www.manning.com/pharrington/)

##Tutorials
###1.) Machine Learning with Scikit-Learn (I) w/ Jake VanderPlas
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


###2.) Machine Learning with Scikit-Learn (II) w/ Olivier Grisel
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
    words that appear at most twice in the dataset. The output is a unique sparse matrix that does NOT store the zeros (ie. compressed). We could use `array.toarray()` or `array.todense()` to bounce between these
representations.
  - TfidfVectorizer(token_pattern=r'(?u)\b[\w-]+\b') treat hyphen as a letter and do not exclude single letter tokens.
<pre>
analyzer = TfidfVectorizer(
    preprocessor=lambda text: text,  # disable lowercasing
    token_pattern=r'(?u)\b[\w-]+\b', # treat hyphen as a letter
                                      # do not exclude single letter tokens
).build_analyzer()

analyzer("I love scikit-learn: this is a cool Python lib!")
</pre>

###3.) Winning Machine Learning Competitions With Scikit-Learn w/ David Chudzicki
David's full presentation is available on github: [ML Comp](https://github.com/dchudz/pycon2015-kaggle-tutorial). 

I use [anaconda](https://store.continuum.io/cshop/anaconda/). So to
start this tutorial, I had to set up a virtual environment using the
command `conda env create` and then activate it using `source activate kaggletutorial`. More details on virtual environments using anaconda [here](http://conda.pydata.org/docs/intro.html#creating-python-3-4-or-python-2-6-environments).

#####2015-04-09 lecture notes:
1. How to focus on quick iteration.
  - First, split the available data (train.csv) into a training set and
    testing set.
  - Decide on feature to engineer (ie. we added title length)
  - Instantiate some models, play with the paramaters
  - Submit score

2. Try it yourself. (my attempt is below. I didn't get last place! =])  
The person who won, Kevin Markham, has an instructional [kaggle blog series](http://blog.kaggle.com/2015/04/08/new-video-series-introduction-to-machine-learning-with-scikit-learn/) on scikit learn with an accompanying [github repo](https://github.com/justmarkham/scikit-learn-videos)
<pre>
  \# My 1st Kaggle Submission
  from sklearn.cross_validation import train_test_split
  from sklearn.linear_model import LogisticRegression
  import numpy as np
  import matplotlib.pyplot as plt
  import seaborn as sns
  import pandas as pd

  %matplotlib inline

  \#load dataset
  train = pd.read_csv("../data/train.csv")

  \# adds length of title as a feature to the dataset
  train["TitleLength"] = train.Title.apply(len)


  train["tagCount"] = (~train.Tag1.isnull()).astype(int) + (~train.Tag2.isnull()).astype(int) + (~train.Tag3.isnull()).astype(int) + (~train.Tag4.isnull()).astype(int) + (~train.Tag5.isnull()).astype(int)
      
  \# split into training and test
  mytrain, mytest = train_test_split(train, test_size = .4)

  \# instantiate model
  lr = LogisticRegression()

  \#fit model
  lr.fit(X=np.asarray(mytrain[["TitleLength","tagCount"]]), y = np.asarray(mytrain.OpenStatus))

  \#predict
  predictions = lr.predict_proba(np.asarray(mytest[["TitleLength","tagCount"]]))[:,1]

  \#compute log loss
  from sklearn.metrics import log_loss
  print(log_loss(mytest.OpenStatus, predictions))

  \#submission
  test = pd.read_csv("../data/test.csv")
  test["tagCount"] = (~train.Tag1.isnull()).astype(int) + (~train.Tag2.isnull()).astype(int) + (~train.Tag3.isnull()).astype(int) + (~train.Tag4.isnull()).astype(int) + (~train.Tag5.isnull()).astype(int)
      
  predictions = lr.predict_proba(np.asarray(test[["ReputationAtPostCreation","tagCount"]]))[:,1]
  submission = pd.DataFrame({"id": test.PostId, "OpenStatus": predictions})
  submission.to_csv("../submissions/fourth_submission.csv", index = False)
  !head ../submissions/fourth_submission.csv
</pre>

###4.) Twitter Network Analysis with NetworkX w/ Sarah Guido, Celia La
Sarah and Celia's full presentation is available on github: [networkx-tutorial](https://github.com/sarguido/networkx-tutorial) or review the [slide deck](http://www.slideshare.net/SarahGuido/network-theory-pycon)


#####2015-04-09 lecture notes:
In order to use the Twitter API, you'll need (see Random notes for
further details or [this site](http://nbviewer.ipython.org/github/furukama/Mining-the-Social-Web-2nd-Edition/blob/master/ipynb/__Chapter%201%20-%20Mining%20Twitter%20%28Full-Text%20Sampler%29.ipynb)):

* import oauth2 (pip install oauth2)
* A twitter account
* Twitter Consumer/Access tokens
* pip install twitter  

Three measures emerge:  

* Degree centrality - **Most edges** == most important  (for directed
  graphs, we can also consider in/out degree centrality)
* Betweenness centrality - **Between the most pairs of nodes** == most
  importnat  
* Closeness centrality - **Average length of shortest paths** == most
  important  

Export for D3:  

* [Export Methods](https://networkx.github.io/documentation/latest/reference/readwrite.json_graph.html)
<pre>
  >>> from networkx.readwrite import json_graph
  >>> G = nx.Graph([(1,2)])
  >>> data = json_graph.node_link_data(G)
</pre>

##Main Sessions 

In general, these talks were much more high level introductions.  

###1.) Machine Learning 101 w/ Kurt Grandis  

- Spectrum: Hancrafted Rules | Statistics | Machine Learning | Deep
  Learning  

- Major ML tools: (K-means, SVM, Random Forrests)  

- Deep Learning (Neural Networks, ect.)  

- Ideans mentioned:  
  - Manifold Hypothesis  
  - Classification - drawing a boundary.  
  - Regression - prediction

- Learning Functions y = f(x|a)
  - Output could be a lable, numeric value

- Common split (80% training, 20% validation)

- Recommendation System  
  - Probabilistic matrix algorithm

###2.) "Words, Words, Words"; Using Python to read Shakespear w/ Adam Palay

- NLTK  
  - [FreqDist](http://www.nltk.org/book/ch01.html#frequency-distributions)
  - Frequency Distribution ([nltk.FreqDist docs](http://www.nltk.org/_modules/nltk/probability.html))  
  - Conditional Frequency Distribution (nltk.ConditionalFreqDist)  
- Classifying  
  - Vectorizer or Feature Extraction  
  - Classifier only interacts w/ teh features  
- How to vectorize  
  - Bag of Words  
  - Sparse matrix  
- Further explanation was relevant to using classifiation  

###3.) Beyond PEP 8 -- Best practices for beautiful intelligible code w/ Raymond Hettinger  
- "Do PEP 8 unto thyself, not unto others."  
- "Treat as a style guide, not a rule book."  
- Unit test, unit test, unit test
- See [docs](https://www.python.org/dev/peps/pep-0008/)

###4.) Distributed Systems 101 w/ [lvh](https://github.com/lvh)  
- [Slides](http://www.lvh.io/DistributedSystems101/#/sec-title-slide)

- Trade-offs:  
  - Availability vs Consistencya
  - Performance	vs Ease of reasoning
  - Scalability	vs Transactionality  

###5.) Grids, Streets and Pipelines: Building a linguistic street map with scikit-learn [repo](https://github.com/michelleful/SingaporeRoadnameOrigins)
 - notes? (didn't attend, but it looked interesting)

###6.) Advanced Git w/ David Baumgold [@singingwolfboy](https://twitter.com/singingwolfboy)
- [Slides](http://bit.ly/git-pycon-2015)
- `git status`
- `git show`  
  - w/out arguments, shows details about current commit
  - w/ argrument, shows details about given commit
- `git blame path/to/file.py`
  - The last commit that touched a line in that file.
- `git cherry-pick commitHash`
  - switch to brach that you want to append the comment that you
    accidentally put on master
  - `git cherry-pick commitHash`
    - creates a new commit (copy of the commitHash)
  - `git reset --hard HEAD^`
    - this will remove the current commit
    - HEAD = latest commit that we have on this branch
    - HEAD^ = parent of latest commit that we have on this branch
- `git rebase`  
  - Mater changed since I started using my branch. I want to bring my
branch up to date with master  
  - `git checkout myBranch`
  - `git rebase master`
  - `git push -f`
- `git reflog`
  - shows commits in the of when you last referenced them
- `git log` 
  - shows commits in ancestor order
- `git rebase --interative HEAD^^^^^` OR `git rebase --interative
  HEAD~5`

###7.) Interactive data for the web - Bokeh for web developers w/ [@birdSarah](https://twitter.com/birdsarah)

Sarah's presentation is avilable in this [repo](https://github.com/birdsarah/pycon_2015_bokeh_talk).

Data visualization using python.  

- great for mid-data (and big-data)
- real-time data updates
- server-side processing

###8.) WebSockets from the Wire Up w/ [@Spang](https://twitter.com/spang)
Christine Spang's presentation will is availble in this [repo](https://github.com/spang/websockets-from-the-wire-up).  
What are websockets?  

- The web was originally reated to share academic documents. The average
  HTTP request is about 800 bytes.  
- AJAX (asynchronous javascript) - potentially update a subset of the
  current page w/out reloading the entire page. Still requires creating a new HTTP request to keep checking in with the server. Communication with the client and server is one way.  
- Websockets open up te communication channel so this HTTP request is
  not being "abused".  

Python Websockets Example:

#####client
<pre>
#
# example from http://aaugustin.github.io/websockets/

import asyncio
import websockets


@asyncio.coroutine
def hello(websocket, path):
    name = yield from websocket.recv()
    print("< {}".format(name))
    greeting = "Hello {}!".format(name)
    yield from websocket.send(greeting)
    print("> {}".format(greeting))

# Normally websockets go over regular HTTP(S) ports (80/443), but we want
# to be able to run this example as non-root, so we use a high-numbered port.
start_server = websockets.serve(hello, 'localhost', 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
</pre>


#####server
<pre>
#
# example from http://aaugustin.github.io/websockets/

import asyncio
import websockets


@asyncio.coroutine
def hello(websocket, path):
    name = yield from websocket.recv()
    print("< {}".format(name))
    greeting = "Hello {}!".format(name)
    yield from websocket.send(greeting)
    print("> {}".format(greeting))

# Normally websockets go over regular HTTP(S) ports (80/443), but we want
# to be able to run this example as non-root, so we use a high-numbered port.
start_server = websockets.serve(hello, 'localhost', 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()

</pre>


###9.) Improve your development environments with virtualization w/ Luke Sneeringer

High level summary: virtualization is good for many reasons.  

About that which I learned while he was talking (my team would probably
call this my 'Golden Retriever Learning Style'):  

##### how to set up a virtural enviroment
- Virtalenv (see this older [repo](https://github.com/lukesneeringer/pycon2013-socketio))
<pre>
mkvirtualenv pycon2013_socketio --python=python2.7
workon pycon2013_socketio
git clone http://github.com/lukesneeringer/pycon2013-socketio.git
cd pycon2013-socketio/
pip install -r pip-requirements.txt
...
deactivate
</pre>
- Anaconda
<pre>
conda create -n testEnv scikit-learn python=2.6 anaconda
source activate testEnv
...
source deactivate
</pre>  


###10.) iPython notebook within Google Docs!
See [colaboratory](http://colaboratory.jupyter.org/welcome/).

##Links
Bayesian stat from Allen Downey:  
  * [Think Bayes](http://www.greenteapress.com/thinkbayes/)  
  * [His other books are here](http://www.greenteapress.com/)  
Visualizations:  
  * [Kaggle Comp Process Visualization](http://datascience.computingpatterns.com/)

##Random notes
0. Conda specific notes:
 
  - Get dependencies:  
  <pre>
  conda depends scikit-lear
  </pre>
  - View currently created environments
<pre>
conda info -e
    Known Anaconda environments:
</pre>

1. Handy ipython tid bits  

  - Get Twitter Data from the public api (**we had problems w/ this in the NetworkX lecture method**)
  
  - I added these details from [this site](http://nbviewer.ipython.org/github/furukama/Mining-the-Social-Web-2nd-Edition/blob/master/ipynb/__Chapter%201%20-%20Mining%20Twitter%20%28Full-Text%20Sampler%29.ipynb):  

<pre>
  import twitter

  \# XXX: Go to http://dev.twitter.com/apps/new to create an app and get values
  \# for these credentials, which you'll need to provide in place of these
  \# empty string values that are defined as placeholders.
  \# See https://dev.twitter.com/docs/auth/oauth for more information 
  \# on Twitter's OAuth implementation.

  CONSUMER_KEY = ''
  CONSUMER_SECRET = ''
  OAUTH_TOKEN = ''
  OAUTH_TOKEN_SECRET = ''

  auth = twitter.oauth.OAuth(OAUTH_TOKEN, OAUTH_TOKEN_SECRET,
                             CONSUMER_KEY, CONSUMER_SECRET)

  twitter_api = twitter.Twitter(auth=auth)

  \# Nothing to see by displaying twitter_api except that it's now a
  \# defined variable

  print twitter_api

  \# The Yahoo! Where On Earth ID for the entire world is 1.
  \# See https://dev.twitter.com/docs/api/1.1/get/trends/place and
  \# http://developer.yahoo.com/geo/geoplanet/

  WORLD_WOE_ID = 1
  US_WOE_ID = 23424977

  \# Prefix ID with the underscore for query string parameterization.
  \# Without the underscore, the twitter package appends the ID value
  \# to the URL itself as a special case keyword argument.

  world_trends = twitter_api.trends.place(_id=WORLD_WOE_ID)
  us_trends = twitter_api.trends.place(_id=US_WOE_ID)

  import json

  \#print json.dumps(world_trends, indent=1)
  print
  \#print json.dumps(us_trends, indent=1)
  world_trends_set = set([trend['name'] 
                          for trend in world_trends[0]['trends']])

  us_trends_set = set([trend['name'] 
                       for trend in us_trends[0]['trends']]) 

  common_trends = world_trends_set.intersection(us_trends_set)

  print common_trends


  \# XXX: Set this variable to a trending topic, 
  \# or anything else for that matter. The example query below
  \# was a trending topic when this content was being developed
  \# and is used throughout the remainder of this chapter.

  q = '#MentionSomeoneImportantForYou' 

  count = 100

  \# See https://dev.twitter.com/docs/api/1.1/get/search/tweets

  search_results = twitter_api.search.tweets(q=q, count=count)

  statuses = search_results['statuses']


  \# Iterate through 5 more batches of results by following the cursor

  for _ in range(5):
      print "Length of statuses", len(statuses)
      try:
          next_results = search_results['search_metadata']['next_results']
      except KeyError, e: # No more results when next_results doesn't exist
          break
          
      \# Create a dictionary from next_results, which has the following form:
      \# ?max_id=313519052523986943&q=NCAA&include_entities=1
      kwargs = dict([ kv.split('=') for kv in next_results[1:].split("&") ])
      
      search_results = twitter_api.search.tweets(**kwargs)
      statuses += search_results['statuses']

  \# Show one sample search result by slicing the list...
  \#print json.dumps(statuses[0], indent=1)


  print statuses[0].keys()
  print
  print statuses[0]["text"]
</pre>
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




