# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:00:44 2019

@author: Suresh Dhamapurkar
"""

import nltk.stem
from sklearn.feature_extraction.text import TfidfVectorizer
"""
import sklearn.datasets
groups = [
    'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware', 'comp.windows.x', 'sci.space']

train_data = sklearn.datasets.fetch_20newsgroups(subset="train",
                                             categories=groups)
"""
class StemmedTfidfVectorizer(TfidfVectorizer):

    def build_analyzer(self):
        english_stemmer = nltk.stem.SnowballStemmer('english')
        
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))