#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 17:46:03 2016

@author: Takanori
"""

import nltk
import math
import InformationRetrieval as ir
from collections import Counter
import numpy as np

docs = [
    ["肉", "寿司", "ピザ", "ラーメン", "肉"],
    ["肉","肉","肉","肉"],
    ["寿司","天麩羅","そば","ラーメン","和食", "中心"],
    ["スイーツ","野菜","スイーツ","野菜", "交互"]
]

"""
#tfidfの表示
tfidf_result = ir.tfidf(docs)
for l in tfidf_result:
    
    print(l)
print(' ')

ridf_result = ir.ridf(docs)
for r in ridf_result:
    
    print(r)
"""

'''
TFを算出する関数

sentences = [
    ["肉", "寿司", "ピザ", "ラーメン", "肉"],
    ["肉","肉","肉","肉"],
    ["寿司","天麩羅","そば","ラーメン","和食", "中心"],
    ["スイーツ","野菜","スイーツ","野菜", "交互"]    
] 
'''
#Term Frequencyを算出する関数
def tf(sentences):
    
    tf_vocab_list = []
    for n in range(len(sentences)):
    
        #print(n)
        #入力された文章をわかち書きしている
        #sentences_wakati_list = tagger.parse(sentences).split() #
        sentences_wakati_list = sentences[n]
        #print(sentences_wakati_list)  
        
        #文章内（その文）での全単語数
        word_count = len(sentences_wakati_list)
        #print('文章内での単語数: '+str(word_count))
        
        #各単語が文章内に何回出てきているかの辞書を作成
        sentences_dict = Counter(sentences_wakati_list)
        #print('各単語の出現頻度の辞書: '+str(sentences_dict))
        
        #TFを辞書形式で保存
        tf_vocab = {}
        for t in sentences_dict.keys():
        
            tf_vocab.update({t : sentences_dict[t]/word_count})
        #print('TFを辞書形式で表した: '+str(tf_vocab))
        tf_vocab_list.append(tf_vocab)
    
    return tf_vocab_list
    
    
sentences = [
    ["肉", "寿司", "ピザ", "ラーメン", "肉"],
    ["肉","肉","肉","肉"],
    ["寿司","天麩羅","そば","ラーメン","和食", "中心"],
    ["スイーツ","野菜","スイーツ","野菜", "交互"]    
] 

res = tf(sentences)

print(res)

# 最初の文章のみ抜き出し
sentence = sentences[0]

# 最初の文章の単語の頻度を算出
sentences_dict = Counter(sentence)

# 文章中の単語の頻度を算出
print(sentences_dict)
# 頻度の最大値を取得
print(max(sentences_dict.values()))
max_value = max(sentences_dict.values())

# 頻度が最大値の単語を取得
print(max(sentences_dict.keys()))

# Augmented Term Frequency
print(0.5+0.5 * (sentences_dict['ラーメン'] / max_value ))

# 重複しない単語の辞書の作成
print(set(sentence))
print(' ')
# Augmented Term Frequencyの関数を作成
def atf(sentences):
    
    # 最終的な結果を返すリスト
    atf_vocab_list = list()
    for i in range(0, len(sentences)):
        
        # 文章中の単語の頻度を算出
        sentence_term_counter = Counter(sentences[i])
        
        # 文章中の単語の最大の頻度を算出
        max_value = max(sentence_term_counter.values())
        
        # 文章中の単語の重複しない辞書の作成
        sentence_term = set(sentences[i])
        """
        print('atfの実行')
        print(sentence_term_counter)
        print(max_value)
        print(sentence_term)
        """
        # atfの計算
        # 1つの文章のatfを算出する
        atf_vocab = {}
        for w in sentence_term:
            atf_vocab.update({w : 0.5 + (0.5 * (sentence_term_counter[w] / max_value))})
                
        atf_vocab_list.append(atf_vocab)
            
    return atf_vocab_list     
        
res = atf(sentences)
print(res)
print(' ')
res = ir.atfidf(sentences)
print(res)







    