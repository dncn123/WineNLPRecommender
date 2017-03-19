
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
from igraph import *
from scipy.spatial.distance import cosine

class WineNetwork(object):
    
    def __init__(self, norminvdf, data):
        self.norminvdf = norminvdf
        self.data = data
        self.wines = norminvdf.shape[0]
        
    def getsimilaritymatrix(self):
        self.similaritymatrix = np.zeros((self.wines,self.wines))
        self.subset = np.invert(np.isnan(self.norminvdf.sum(axis=1)))
        for r in range(self.wines):
            for c in range(self.wines):
                if r <= c:
                    if self.subset[r] & self.subset[c]:
                        self.similaritymatrix[r,c] = cosine(self.norminvdf[r],self.norminvdf[c])
                    else:
                        self.similaritymatrix[r,c] = 1
                else:
                    self.similaritymatrix[r,c] = self.similaritymatrix[c,r]
        self.similaritymatrix
    
    def makegraph(self, colrange, simlimit, edgemethod="limit", top=5):
        self.g = Graph()
        self.g.add_vertices(self.data["wine_id"].values)
        for col in self.data.columns[colrange]:
            self.g.vs[col] = self.data[col].values
        if edgemethod == "limit":
            self.g.add_edges([(r,c) for r in range(self.wines) 
                              for c in range(self.wines) if (r < c) & (self.similaritymatrix[r,c]<simlimit)])
        elif edgemethod == "top":
            edges = []
            for i in range(self.wines):
                indexes = self.similaritymatrix[i].argsort()[1:top+1]
                indexes = indexes[self.similaritymatrix[i][indexes]<simlimit]
                for j in indexes:
                    if i < j:
                        edges.append((i, j))
                    else:
                        edges.append((j, i))
            self.g.add_edges(list(set(edges)))
            
            
    def plotgraph(self):
        cols = ["red", "white", "yellow", "pink", "brown", "cyan", "black"]
        col_dict = {col:cols[i] for i, col in enumerate(self.data.Colour.unique())}
        visual_style = {}
        visual_style["vertex_size"] = 7
        visual_style["edge_width"] = 0.25
        visual_style["vertex_color"] = [col_dict[c] for c in self.g.vs["Colour"]]
        layout = self.g.layout("fr")
        return plot(self.g, layout=layout, **visual_style)

