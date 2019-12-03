from flask import Flask, render_template, request, redirect, url_for
from flask_pymongo import PyMongo

app = Flask(__name__)
mongo = PyMongo(app, uri ="mongodb://localhost:27017/project")

@app.route('/', methods=['POST','GET'])

def index():
    task_content=''
    if request.method == 'POST':
        task_content = request.form['content']
        task = mongo.db.graph.aggregate([ {'$match':{ 'node':task_content }}, {'$sort':{ 'weight':-1 }}, {'$limit':30 }, {'$lookup':{ 'from':"degree", 'let':{'checker':"$node"}, 'pipeline': [ {'$match':{'$expr':{'$eq':["$_id","$$checker"]}}}, {'$project':{'_id':0}} ], 'as': "degree" }}, {'$sort':{ 'degree':-1 }}, {'$project':{ '_id':0,'arc':1 } } ])
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