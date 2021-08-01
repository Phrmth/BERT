import pandas as pd
from sklearn import metrics,preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model as lm

# Reading train and test data

train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
y = train.target


# Extracting text into one

traindata = list(np.array(train.text))
testdata = list(np.array(test.text))
X_all = traindata + testdata
lentrain = len(traindata)

# Creating TF-IDF vector and LR model

tfv = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',  
        analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1)

rd = lm.LogisticRegression(penalty='l2', dual=False, tol=0.0001, 
                             C=1, fit_intercept=True, intercept_scaling=1.0, 
                             class_weight=None, random_state=None)

tfv.fit(X_all)
X_all = tfv.transform(X_all)
X = X_all[:lentrain]
X_test = X_all[lentrain:]

# Testing cross validation 
print(cross_val_score(rd, X, y, cv=20, scoring='roc_auc'))

rd.fit(X,y)

# predicting on test data

pred = rd.predict(X_test)




