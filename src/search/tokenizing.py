import sys
import os

import pandas as pd
import numpy as np

import nltk

import konlpy
from konlpy.tag import *
from konlpy.utils import pprint

sentence = ""
tokenizing = ""
model = ""
h = ""

h = isHangul(sentence)

if h&(tokenizing == "refined"):
    print("한국어/refined...")
    checklist = kor_demo
    kkma = Kkma()
    kk_pos = kkma.pos(sentence)
    kk_morph = kkma.morphs(sentence)

    lst = ["NNG","NNP","NNB","NNM","NR","NP","VV","VA","VXV","VXA","VCP","VCN"]

    indice = np.array( [ 1 if v in lst else 0 for (k,v) in kk_pos] )
    temp = list(np.array(kk_morph)[np.array(indice,dtype = bool)])
    print(temp)
    if model == 'w2v':
        print('w2v')
        test = w2v_seq2vec(temp,kor_w2v_model)
        cmp_lists = list(kor_demo['r_w2v_vec'])
    else:
        print('fastText')
        test = ft_seq2vec(temp,d)
        cmp_lists = list(kor_demo['r_ft_vec'])        

    print(test)
    
elif h&(tokenizing != "refined"):
    print("한국어/full...")
    checklist = kor_demo
    kkma = Kkma()
    kk_morph = kkma.morphs(sentence)
    print(kk_morph)
    
    if model == 'w2v':
        print('w2v')
        test = w2v_seq2vec(kk_morph,kor_w2v_model)
        cmp_lists = list(kor_demo['w2v_vec'])
    else:
        print('fastText')
        test = ft_seq2vec(temp,d)
        cmp_lists = list(kor_demo['ft_vec'])        

    print(test)
    
elif (h==False) &(tokenizing == "refined"):
    print("영어/refined...")
    checklist = eng_demo
    nltk_morph = nltk.word_tokenize(sentence)
    nltk_tag = nltk.pos_tag(nltk_morph)

    lst = ["NN","NNP","NNPS","NNS","PRP","PRP$","VB","VBD","VBG","VBN","VBP","VBZ"]

    nltk_indice = np.array( [ 1 if v in lst else 0 for (k,v) in nltk_tag] )
    nltk_temp = list(np.array(nltk_morph)[np.array(nltk_indice,dtype = bool)])
    print(nltk_temp)
    test = w2v_seq2vec(nltk_temp,eng_w2v_model)
    cmp_lists = list(eng_demo['r_w2v_vec'])
    print(test)
    
else:
    print("영어/full...")
    checklist = eng_demo
    nltk_morph = nltk.word_tokenize(sentence)
    print(nltk_morph)
    test = w2v_seq2vec(nltk_morph,eng_w2v_model)
    cmp_lists = list(eng_demo['w2v_vec'])
    print(test)