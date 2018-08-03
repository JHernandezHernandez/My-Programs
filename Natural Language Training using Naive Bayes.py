#Natural Language Processing using Naive Bayes

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
# delimiter \t to adjust for .tsv file. quoting; quoting = 3 ignores double quotes

# Clearning the texts
import re # For getting rid of everything except what is specified here.
import nltk # Natural Language ToolKit
nltk.download('stopwords') # Downloads set of words we can generally eliminate
from nltk.corpus import stopwords #So that you don't have warnings from stopwords. 
# Need to run this line too
from nltk.stem.porter import PorterStemmer # For stem of words

corpus = [] # Seems to creat an empty list
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])# Only keep letters
    # The space is for letting the [a-zA-Z] to work so it doesn't get messed up.
    # Only applicable to the first(review) line of the file.
    review = review.lower()
    # everything will be lowercase
    review = review.split() 
    # Split words so that we have a list instead
    ps = PorterStemmer() # Creating object of PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] # For loop; set is faster.
    review = ' '.join(review) #joins words together with a space in between
    corpus.append(review)
    
# Creating the BAg of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) # Most common words. Gets rid of 65 least frequent
X = cv.fit_transform(corpus).toarray() #Creates matrix from corpus variable
y = dataset.iloc[:, 1].values #take index of column

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# The resulting computing matrix in cm goes as follows ('row', 'column'):
# The (0,0) entry means it made that many correct predictions of negative reviews. True Negative
# The (1,1) entry means it made that many correct predictions of positive reviews. True Positive 
# The (0,1) entry means it made that many incorrect preditions of positive reviews. False Positive
# The (1,0) entry means it made that many incorrect preditions of negative reviews. False Negative

(54 + 87)/(200) # Correct Preditions