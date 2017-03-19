
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import re
from collections import Counter
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
import pickle
stop = stopwords.words('english')


# In[41]:

class preprocess(object):
    
    def __init__(self, data):
        self.data = data
        self.wines = len(data.index)
        
    def extracttext(self, description):
        if type(description)!=str:
            return []
        text = re.sub(r'[^\w\s]','',description.lower())
        text = re.sub('  ',' ',text)
        textlist = [word.strip() for word in text.split(' ') if word.strip() not in stop]
        return textlist + [textlist[i-1]+' '+textlist[i] for i in range(1,len(textlist))]
    
    def gettextbag(self):
        self.text_bag = np.array([self.extracttext(d) for d in self.data.Description])
        
    def mergesamewords(self, matchamount = 90):
        wordconvert = {}
        all_text_bag = [word for wordlist in self.text_bag for word in wordlist if len(word)>1]
        all_text_bag.sort()
        all_text_bag = np.unique(all_text_bag)
        for i1, term1 in enumerate(all_text_bag):
            for i2, term2 in enumerate(all_text_bag):
                if i1 <= i2:
                    continue
                elif term2 in wordconvert:
                    continue
                elif term1[0] != term2[0]:
                    continue            
                elif abs(len(term1)-len(term2)) > 3:
                    continue
                elif fuzz.ratio(term1, term2) > matchamount:
                    wordconvert[term2] = term1
        for i, text in enumerate(self.text_bag):
            self.text_bag[i] = [wordconvert[term] if term in wordconvert else term for term in text]
        self.text_bag = np.array(self.text_bag)
    
    def selecttermstokeep(self, lowercut=1):
        all_text_bag = [word for wordlist in self.text_bag for word in wordlist]
        c = Counter()
        c.update(all_text_bag)
        self.termstokeep = [k for k, v in c.items() if (v > lowercut) & (len(k)>1)]
        for i, text in enumerate(self.text_bag):
            self.text_bag[i] = [terms for terms in text if terms in set(self.termstokeep)]
        self.text_bag = np.array(self.text_bag)
    
    def maketermdict(self):
        self.termdict = {}
        for i, text in enumerate(self.text_bag):
            for term in self.termstokeep:
                if term not in set(text):
                    continue
                if term not in self.termdict:
                    self.termdict[term] = {i: 1}
                else:
                    self.termdict[term][i] = 1 # not counting number of just presense 
    
    def maketermmatrix(self):
        self.termmatrix = np.zeros((len(self.termdict),self.wines))
        dictindex = {}
        index = 0
        for d in self.termdict:
            dictindex[index] = d
            index += 1
        for i, termrow in enumerate(self.termmatrix):
            for j in range(len(termrow)):
                if j in self.termdict[dictindex[i]]:
                    termrow[j] = 1
        self.termmatrix = self.termmatrix.T
    
    def removenondiffwords(self, grouping="Colour", size = 50):
        bigenough = self.termmatrix.sum(axis=0)>size
        variation = np.array([list(self.termmatrix[(self.data[grouping] == g).values,:].sum(axis=0)/sum((self.data[grouping] == g))) 
                              for g in self.data[grouping].unique()]).std(axis=0)
        medianvariation = np.median(variation[bigenough])
        boole = zip(variation>medianvariation,bigenough)
        subset = np.array([True if i and j else True if not j else False for i, j in boole])
        self.termmatrix = self.termmatrix[:,subset]
    
    def makeinversedtmatrix(self):
        invdf = np.log(self.wines/self.termmatrix.sum(axis=0))
        unnormtfinvdf = (self.termmatrix*invdf)
        self.norminvdf = (unnormtfinvdf.T/np.sqrt(np.power(unnormtfinvdf,2).sum(axis=1)).T).T
        self.norminvdf[np.isnan(self.norminvdf.sum(axis=1))]=0
        return self.norminvdf
    
    def preprocessdatting(self, grouping="Colour", size=50, matchamount = 90, lowercut=1):
        self.gettextbag()
        self.mergesamewords(matchamount = matchamount)
        self.selecttermstokeep(lowercut=lowercut)
        self.maketermdict()
        self.maketermmatrix()
        self.removenondiffwords(grouping=grouping, size=size)
        return self.makeinversedtmatrix()

