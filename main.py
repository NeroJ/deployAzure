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


UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = set(['csv'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def predictLable(modelType = 'LRlbfgs', filename = 'test.csv'):
    downloadName = filename+'_'+modelType +'_predicted.csv'
    df = pd.read_csv("D:\home\site\wwwroot\uploads" + filename)
    dfList = np.array(df)
    length_ = len(dfList)
    alpha = 0.01
    one_Num = int(length_ * alpha)
    randomList = [random.randint(0,length_) for i in range(one_Num)]
    result = []
    for i in range(length_):
        if i in randomList:
            result.append(1)
        else:
            result.append(0)
    df.insert(len(list(df.columns)), 'Result', np.array(result))
    #-----to.csv-----#
    df.to_csv("D:\home\site\wwwroot\uploads"+ downloadName)


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
            predictLable(modelType = modelType, filename = filename)
            time.sleep(5.5)
            return redirect(url_for('download_file',
                                    filename=filename+'_'+modelType +'_predicted.csv'))
    return render_template('hello.html')

@app.route('/downloads/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename,as_attachment=True)

if __name__ == '__main__':
  app.run()
