# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 12:32:36 2019

@author: Suresh Dhamapurkar
"""
import pickle
import numpy as np
from datetime import datetime
import scipy as sp
from classTifdf import StemmedTfidfVectorizer
from flask import Flask, render_template, request, jsonify
#import requests

app = Flask(__name__)

@app.route('/knowledge_post', methods=['POST'])
def predict_post_cluster():
    new_post = request.form['know_post']
    filename_km = "news-post-cluster-model" +  '.pkl'
    print(filename_km)
    with open(filename_km, 'rb') as model_file:
        model_km = pickle.load(model_file)

    #filename_vect = r"C:\00_Users\sdd\00_GD2\bzn\prj\fima\Src\python\TextAnal\vectorizer-" +  datetime.now().strftime("%Y%m%d").upper() + '.pkl'
    filename_vect = "vectorizer" + '.pkl'
    print(filename_vect)
    with open(filename_vect, 'rb') as vect_file:
        vectorizer = pickle.load(vect_file)
    #vectorized = vectorizer.fit_transform(train_data.data)   
    #filename_vected = r"C:\00_Users\sdd\00_GD2\bzn\prj\fima\Src\python\TextAnal\vectorized-" +  datetime.now().strftime("%Y%m%d").upper() + '.pkl'
    filename_vected = "vectorized" + '.pkl'
    with open(filename_vected, 'rb') as vected_file:
        vectorized = pickle.load(vected_file)
    #filename_data = r"C:\00_Users\sdd\00_GD2\bzn\prj\fima\Src\python\TextAnal\train_data-" +  datetime.now().strftime("%Y%m%d").upper() + '.pkl'
    filename_data = "train_data" + '.pkl'
    with open(filename_data, 'rb') as data_file:
        train_data = pickle.load(data_file)
    print (type(train_data))    
    #new_post_file = request.files.get("input_file")
    #new_post = new_post_file.read()
    #print (new_post)
    print ("Decoded string printed-----------")
    #print (str(new_post,"utf-8") )
    print (type(vectorizer))
    #vectorizer._validate_vocabulary()
    new_post_vec = vectorizer.transform([new_post])
    #print (new_post_vec)
    #new_post_vec = new_post_vec.decode("utf-8") 
    new_post_label = model_km.predict(new_post_vec)[0]
    print ("New post cluster:", new_post_label )
    similar_indices = (model_km.labels_ == new_post_label).nonzero()[0]
    #print (similar_indices)
    print (type(similar_indices))
    print (type(str(similar_indices)))
    similar = []
    for i in similar_indices:
        dist = sp.linalg.norm((new_post_vec - vectorized[i]).toarray())
        similar.append((dist, train_data.data[i]))
    
    similar = sorted(similar)
    print("Count similar: %i" % len(similar))
    cats = []
    for i in similar_indices:
        k = train_data.target[i]
        cats.append(train_data.target_names[k])
    print (len(cats))
    print (set(cats))

    show_at_1 = similar[0]
    show_at_2 = similar[1] #similar[int(len(similar) / 10)]
    show_at_3 = similar[2] #similar[int(len(similar) / 2)]
    show_median = similar[int(len(similar) / 2)]
    show_last = similar[-1] 

    resp1 = str(list(similar_indices))
    #return {'new_post_label':new_post_label, 'resp1': resp1}
    print ("Hiiiiiii There !!!!")
    data = {'new_post_label':int(new_post_label), 'show_at_1':show_at_1, 'show_at_2':show_at_2, 'show_at_3':show_at_3, 'zresp1': resp1}
    print (jsonify({'new_post_label':int(new_post_label), 'show_at_1':show_at_1, 'show_at_2':show_at_2, 'show_at_3':show_at_3, 'resp1': resp1}))
    #return jsonify({'new_post_label':int(new_post_label), 'show_at_1':show_at_1, 'show_at_2':show_at_2, 'show_at_3':show_at_3, 'resp1': resp1})
    

    
    
    # [14-Aug-2019 Suresh] Using own key
    # r = requests.get('http://api.openweathermap.org/data/2.5/weather?zip='+zipcode+',us&appid=fd38d62aa4fe1a03d86eee91fcd69f6e')
    #r = requests.get(
        #'http://api.openweathermap.org/data/2.5/weather?zip=' + zipcode + ',in&appid=a8bb1f2d31f37830f715b70df71700f5')
    #json_object = r.json()
    #temp_k = float(json_object['main']['temp'])
    #temp_f = (temp_k - 273.15) * 1.8 + 32
    return render_template('knowledge.html', temp=new_post_label, categories=set(cats), orig_post=new_post, artcle1=show_at_1, cat0=cats[0], artcle2=show_at_2,cat1=cats[1], artcle3=show_at_3, cat2=cats[2], artcle4=show_median, cat3=cats[int(len(similar) / 2)], artcle5=show_last, cat4=cats[-1], list1=resp1 )

@app.route('/')
def index():
	return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)