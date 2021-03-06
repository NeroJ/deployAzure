import sys
import os
import time
import json
import datetime
import numpy as np
import pandas as pd
import random

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
