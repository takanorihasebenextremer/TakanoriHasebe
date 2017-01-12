#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 17:46:03 2016

@author: Takanori
"""

import nltk
import math
import InformationRetrieval as ir


docs = [
    ["肉", "寿司", "ピザ", "ラーメン", "肉"],
    ["肉","肉","肉","肉"],
    ["寿司","天麩羅","そば","ラーメン","和食", "中心"],
    ["スイーツ","野菜","スイーツ","野菜", "交互"]
]

#tfidfの表示
tfidf_result = ir.tfidf(docs)
for l in tfidf_result:
    
    print(l)
print(' ')

ridf_result = ir.ridf(docs)
for r in ridf_result:
    
    print(r)





    