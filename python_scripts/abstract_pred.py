# -*- coding: utf-8 -*-

######################ingest data for abstract to fos prediction############################

import pandas as pd
abstract_fos_df = pd.read_csv(r'abstract_predict.csv',encoding='utf-8')

######################preprocessing for X###################################################

import re

X = abstract_fos_df['indexed_abstract.InvertedIndex']
X = X.apply(lambda x: re.sub('[^A-Za-z, +]', '', x))
X = X.apply(lambda x: [part.strip() for part in x.split(',') if part != ''])

#removing stopwords using nltk

from nltk.corpus import stopwords
stop=stopwords.words('english')

X = X.apply(lambda x: ' '.join([word for word in x if word.lower() not in stop]))

########################preprocessing for y##################################################

y = abstract_fos_df['fos']
y = y.apply(lambda x: re.sub('[^A-Za-z, +]', '', x))
y = y.apply(lambda x: re.findall(r'name([\w ]+),w',x))

#######################generating multiple output columns#################################### 

from sklearn.preprocessing import MultiLabelBinarizer
mlb2 = MultiLabelBinarizer(sparse_output=True)
y_sparse = mlb2.fit_transform(y)
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y)
y = pd.DataFrame(y,columns=mlb.classes_)

######################convert abstract to hashed features#####################################

from sklearn.feature_extraction.text import HashingVectorizer
hv = HashingVectorizer(n_features=1000)
X = hv.transform(X.tolist())

#########################reducing no of output cols to topics with more than 10 occurances

freq = y_sparse.getnnz(axis=0)
cols = list(mlb2.classes_)
in_vals = [cols[i] for i in range(len(cols)) if freq[i] >=50]
y = y[in_vals]

#####################generate train test split###############################################

from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state=2018)
#####################multi output classification###############################

from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier

clf = MultiOutputClassifier(LinearSVC(),n_jobs=7)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

y_pred_df = pd.DataFrame(y_pred,columns=y_test.columns)

#####################accuracy score############################################

from sklearn.metrics import accuracy_score

acc = []

for i in range(len(in_vals)):
    val = accuracy_score(y_test.iloc[:,i],y_pred_df.iloc[:,i])
    #print(i,val)
    acc.append(val)

from statistics import mean 
mean_acc = mean(acc)
print("mean accuracy is",mean_acc)

######################pickles##################################################

import pickle

classifier = open('classifier','wb')
hash_maker = open('hash_maker','wb')
pickle.dump(hv,hash_maker)
pickle.dump(clf,classifier)
classifier.close()
hash_maker.close()

td = open('topic_data.txt','w')
td.writelines(["%s\n" % item for item in list(y_pred_df.columns)])
td.close()        
