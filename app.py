from flask import Flask, render_template, request, redirect, url_for
from flask_pymongo import PyMongo
try:
    from cfuzzyset import cFuzzySet as FuzzySet
except ImportError:
    from fuzzyset import FuzzySet

app = Flask(__name__)
mongo = PyMongo(app, uri ="mongodb://localhost:27017/project")

c_file = open('corpus_file','rb')
o_file = open('origin_file','rb')
t_file = open('terms_file','rb')

import pickle

corpus = pickle.load(c_file)
origin = pickle.load(o_file)
terms = FuzzySet()
terms = pickle.load(t_file)

c_file.close()
o_file.close()
t_file.close()

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


if __name__=="__main__":
    app.run(debug=True)
