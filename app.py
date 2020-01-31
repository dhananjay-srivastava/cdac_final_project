from flask import Flask, render_template, request, redirect, url_for
from flask_pymongo import PyMongo

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop=stopwords.words('english')

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://admin:system123@project-shard-00-00-aavo9.gcp.mongodb.net:27017,project-shard-00-01-aavo9.gcp.mongodb.net:27017,project-shard-00-02-aavo9.gcp.mongodb.net:27017/project?ssl=true&replicaSet=project-shard-0&authSource=admin&retryWrites=true&w=majority"
mongo = PyMongo(app)

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

r_file = open('rule_df2','rb')

import pickle

corpus = pickle.load(c_file)
origin = pickle.load(o_file)
terms = pickle.load(t_file)
hv = pickle.load(hv_file)
clf = pickle.load(clf_file)
rules = pickle.load(r_file)

c_file.close()
o_file.close()
t_file.close()
hv_file.close()
clf_file.close()
r_file.close()



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

def generate_fpg_predictions(topic, rules=rules):
        out = rules[ (rules.antecedents == {topic}) & (rules.consequents_len == 1) ].sort_values(by='confidence',ascending=False).head(20)
        out['consequents'] = out['consequents'].apply(lambda x: list(x)[0]).astype("unicode")
        return out[['consequents','confidence']].values.tolist()
     
@app.route('/', methods=['POST','GET'])

def index():
    if request.method == 'POST':
        form_content=request.form['content']
        if form_content not in origin and form_content != None:
            task = {}
            if form_content.lower() in corpus:
                search_corrected = origin[corpus.index(form_content.lower())]
                task = mongo.db.topic_data.find({'_id':search_corrected},{'_id':0})
                return render_template('index.html', tasks = task)
            else:
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
    task = mongo.db.paper_data.find({'_id':search})
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

@app.route('/fpgrowth',methods=['POST','GET'])

def fpgrowth():
    if request.method == 'POST':
        search = request.form['fpg_content']
        task = {}
        task["topics"] = generate_fpg_predictions(search)
        return render_template('fpgrowth.html', tasks = task)
    else:
        return render_template('fpgrowth.html', tasks = [])
        
if __name__=="__main__":
    app.run(debug=True)
