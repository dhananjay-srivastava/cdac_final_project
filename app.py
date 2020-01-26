from flask import Flask, render_template, request, redirect, url_for
from flask_pymongo import PyMongo


app = Flask(__name__)
mongo = PyMongo(app, uri ="mongodb://localhost:27017/project")

c_file = open('corpus_file','rb')
o_file = open('origin_file','rb')

try:
    from cfuzzyset import cFuzzySet as FuzzySet
except ImportError:
    from fuzzyset import FuzzySet
t_file = open('terms_file','rb')

from sklearn.feature_extraction.text import HashingVectorizer
hv_file = open('hash_maker','rb')

from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
clf_file = open('classifier','rb')

import pickle

corpus = pickle.load(c_file)
origin = pickle.load(o_file)
terms = pickle.load(t_file)
hv = pickle.load(hv_file)
clf = pickle.load(clf_file)

c_file.close()
o_file.close()
t_file.close()
hv_file.close()
clf_file.close()

from nltk.corpus import stopwords
stop=stopwords.words('english')

topics = []
with open(r'F:\topic_data.txt','r') as file:
    for line in file:
        topics.append(line[:-1])

import re
import pandas as pd
def generate_predictions(abstract,stop=stop,topics=topics):
	abstract = re.sub('[^A-Za-z ]', '', abstract)
	out = [' '.join([word for word in abstract.split() if word.lower() not in stop])]
	X = hv.transform(out)
	y_pred = clf.predict(X)
	y_pred_df = pd.DataFrame(y_pred,columns=topics)
	return list(y_pred_df.columns[(y_pred_df == 1).iloc[0]])

@app.route('/', methods=['POST','GET'])

def index():
    if request.method == 'POST':
        form_content=request.form['content']
        if form_content not in origin and form_content != None:
            task = {}
            task['err'] = origin[corpus.index(terms.get(form_content.lower())[0][1])]
            return render_template('index.html', tasks = task)
        else:
            task = mongo.db.topic_data.find({'_id':form_content},{'_id':0})
            return render_template('index.html', tasks = task)
    elif request.method == 'GET':
        dym_content=request.args.get('val2')
        task = mongo.db.topic_data.find({'_id':dym_content},{'_id':0})
        return render_template('index.html', tasks = task)
    else:
        return render_template('index.html', tasks = [])
    
@app.route('/paper',methods=['POST','GET'])

def paper():
    search = request.args.get('val')
    task = mongo.db.paper.find({'_id':search})
    return render_template('paper.html', tasks = task)

@app.route('/abstract',methods=['POST','GET'])

def abstract():
    if request.method == 'POST':
        search = request.form['abs_content']
        task = {}
        task["topics"] = generate_predictions(search)
        return render_template('abstract.html', tasks = task)
    else:
        return render_template('abstract.html', tasks = [])

if __name__=="__main__":
    app.run(debug=True)
