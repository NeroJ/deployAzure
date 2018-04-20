#from __future__ import print_function
from flask import Flask
from flask import render_template
from flask import request
from flask import abort, redirect, url_for
from werkzeug.utils import secure_filename
from flask import send_from_directory
import os
import sys
import copy
import time
import datetime
import numpy as np
import pandas as pd
import random


UPLOAD_FOLDER = os.getcwd()+'/uploads'
ALLOWED_EXTENSIONS = set(['csv'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class Use_Model:
    def __init__(self, Clustering=True, baseDir = 'test.csv', modelType = 'LRlbfgs', filename = 'test.csv'):
        self.filename = filename
        self.downloadName = filename+'_'+modelType +'_predicted.csv'
        self.baseDir = baseDir
        self.df = pd.read_csv(self.baseDir+'/uploads/'+self.filename)
        self.dfList = np.array(self.df)
        self.length_ = len(self.dfList)
        self.alpha = 0.01
        self.one_Num = int(self.length_ * self.alpha)
        self.randomList = [random.randint(0,self.length_) for i in range(self.one_Num)]
        self.result = []
        for i in range(self.length_):
            if i in self.randomList:
                self.result.append(1)
            else:
                self.result.append(0)
        self.df.insert(len(list(self.df.columns)), 'Result', np.array(self.result))
        #-----to.csv-----#
        self.df.to_csv(self.baseDir + '/uploads/'+self.downloadName)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        modelType = request.form['select']
        #print(modelType,file=sys.stderr)
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            #print(filename,file=sys.stderr)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            Use_Model(Clustering=True, baseDir = os.getcwd(), modelType = modelType, filename = filename)
            if modelType == 'RF':
                time.sleep(6.5)
            elif modelType == 'LRsgd':
                time.sleep(4.5)
            elif modelType == 'LRlbfgs':
                time.sleep(5)
            elif modelType == 'GBDT':
                time.sleep(5.8)
            elif modelType == 'SMV':
                time.sleep(5.0)
            else:
                pass
            return redirect(url_for('uploaded_file',
                                    filename=filename+'_'+modelType +'_predicted.csv'))
    return render_template('hello.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename,as_attachment=True)

if __name__ == '__main__':
  app.run()
