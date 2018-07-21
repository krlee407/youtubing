import konlpy
from konlpy.tag import *
from konlpy.utils import pprint

import nltk

import sys
import os
import io
import pandas as pd
import numpy as np
import pickle
import json

from ast import literal_eval
from itertools import chain
import math

from gensim.models import Word2Vec
from sklearn.metrics.pairwise import *

import sqlite3

model = sys.argv[1]

con = sqlite3.connect('data/youtubing.db')

#파일 불러오기
sql_stc = "SELECT \
                A.sentence_id as sentence_id, \
                A.start_time as start_time, \
                A.end_time as end_time, \
                A.sentence as subtitle, \
                A.text_token as text_token, \
                A.embedding_vector as embedding_vector, \
                A.subtitle_id as subtitle_id, \
                B.filename as filename, \
                B.language as lan, \
                B.is_auto_generated as is_auto_generated, \
                B.video_id as video_id \
           FROM sentence_meta as A JOIN subtitle_meta as B ON A.subtitle_id = B.subtitle_id "
demo = pd.read_sql(sql_stc, con)

#demo['morphs'] = demo.apply(lambda row:literal_eval(row['morphs']), axis=1)
demo['text_token'] = demo.apply(lambda row:literal_eval(row['text_token']), axis=1)

#kor/eng 나누기
kor_demo = demo[demo['lan'] == "korean"]
eng_demo = demo[demo['lan'] == "english"]
eng_demo = eng_demo.reset_index()

#kor_morphs = kor_demo['morphs']
kor_refined_morphs = kor_demo['text_token']

#eng_morphs = eng_demo['morphs']
eng_refined_morphs = eng_demo['text_token']

# 2. train word2vec/ft model
if model == 'w2v':
    print("train&save word2vec model...")

    print(" 1.train korea w2c model ...")
    kor_w2v_model = Word2Vec(kor_refined_morphs, size=300, min_count=2, workers=4, sg=1)
    kor_w2v_model.train(kor_refined_morphs, total_examples=len(kor_refined_morphs), epochs=10)
    kor_w2v_model.save("data/kor_w2v_model.model")
    
    print(" 2.train english w2v model ...")
    eng_w2v_model = Word2Vec(eng_refined_morphs, size=300, min_count=2, workers=4, sg=1)
    eng_w2v_model.train(eng_refined_morphs, total_examples=len(eng_refined_morphs), epochs=10)
    eng_w2v_model.save("data/eng_w2v_model.model")

    print("complete.\n")

elif model == 'ft':
    
    print("train&save fastText model...")

    print(" 1. load pretrained korean fastText model...")
    pkl_file = open('data/ft_vec1.pkl', 'rb')
    mydict1 = pickle.load(pkl_file)
    pkl_file.close()

    pkl_file2 = open('data/ft_vec2.pkl', 'rb')
    mydict2 = pickle.load(pkl_file2)
    pkl_file2.close()

    d = {}
    for k,v in mydict1.items():
        d[k] = v
        
    for k,v in mydict2.items():
        d[k] = v
    
    print(" 2.train english w2v model ...")
    eng_w2v_model = Word2Vec(eng_refined_morphs, size=300, min_count=2, workers=4, sg=1)
    eng_w2v_model.train(eng_refined_morphs, total_examples=len(eng_refined_morphs), epochs=10)
    eng_w2v_model.save("data/eng_w2v_model.model")

    print("complete.\n")

else:
    print("train&save fastText & w2v model...")

    print(" 1.train korea w2c model ...")
    kor_w2v_model = Word2Vec(kor_refined_morphs, size=300, min_count=2, workers=4, sg=1)
    kor_w2v_model.train(kor_refined_morphs, total_examples=len(kor_refined_morphs), epochs=10)
    kor_w2v_model.save("data/kor_w2v_model.model")

    print(" 2.train english w2v model ...")
    eng_w2v_model = Word2Vec(eng_refined_morphs, size=300, min_count=2, workers=4, sg=1)
    eng_w2v_model.train(eng_refined_morphs, total_examples=len(eng_refined_morphs), epochs=10)
    eng_w2v_model.save("data/eng_w2v_model.model")

    print(" 3. load pretrained korean fastText model...")
    pkl_file = open('data/ft_vec1.pkl', 'rb')
    mydict1 = pickle.load(pkl_file)
    pkl_file.close()

    pkl_file2 = open('data/ft_vec2.pkl', 'rb')
    mydict2 = pickle.load(pkl_file2)
    pkl_file2.close()

    d = {}
    for k,v in mydict1.items():
        d[k] = v
        
    for k,v in mydict2.items():
        d[k] = v

    print("complete.\n")
    
# 3. tf_idf
#tf_idf를 위해 각 비디오별 자막 생성
print("get tf idf weight...")
def get_subtitle_per_v(demo):
    tt = []
    subtitle_id_list = []
    for sid in set(demo['subtitle_id'].values):
        tmp_demo = demo[demo['subtitle_id'] == sid]
        tt.append(chain.from_iterable(tmp_demo['text_token'].values))
        subtitle_id_list.append(sid)

    subtitle2 = pd.DataFrame({'subtitle_id' : subtitle_id_list, 'text_token' : tt})
    return subtitle2

kor_subtitle2 = get_subtitle_per_v(kor_demo)
eng_subtitle2 = get_subtitle_per_v(eng_demo)

# word당 tf_idf 계산
# def get_w(demo,subtitle2,col,start_v_id):

#     #vocab
#     lst = subtitle2[col].tolist()
#     vocab = list(set(list(chain.from_iterable(lst))))
    
#     #bow(tf)
#     t = pd.DataFrame({'vacab':vocab})
#     sub = pd.DataFrame(subtitle2)

#     def f(subtitle2,voca):

#         def t(morphs,voca):
#             return sum(np.array(morphs) == voca)

#         result = sub.apply(lambda row: t(row[col],voca),axis = 1)
#         return list(result.values)

#     test = t.apply(lambda word: f(subtitle2,word.values),axis = 1)
#     bow = dict( (k,v) for k,v in zip(vocab,test))
    
#     #idf
#     idf = {}
#     N = len(list(bow.values())[0])

#     for k,v in bow.items():
#         if type(np.array(v) == 0) == bool:
#             df = 1 if np.array(v) == 0 else 0
#         else:
#             df = sum(np.array(v) == 0)
#         df = N - df 
#         result = math.log(N / df)
#         idf[k] = result

#     #weight
#     weight = [] 
#     for num,v in enumerate(demo[col]):
#         result = []
#         for element in v:    
#             result.append(bow[element][demo['v_id'][num]-start_v_id]*idf[element])

#         weight.append(result)
    
#     for i in range(len(weight)):
#         w_sum = sum(weight[i])
#         if w_sum == 0:
#             w_sum = 1
#         weight[i] = list(np.array(weight[i])/w_sum)

#     return weight

# word당 tf_idf 계산

kor_demo['v_id'] = list(np.arange(len(kor_demo)))
eng_demo['v_id'] = list(np.arange(len(eng_demo)) + len(kor_demo) -1)

def get_w(demo,subtitle2,col,start_v_id):

    #vocab
    lst = subtitle2[col].tolist()
    vocab = list(set(list(chain.from_iterable(lst))))

    #bow(tf)
    t = pd.DataFrame({'vacab':vocab})
    sub = pd.DataFrame(subtitle2)

    def f(subtitle2,voca):

        def t(morphs,voca):
            return sum(np.array(morphs) == voca)

        result = sub.apply(lambda row: t(row[col],voca),axis = 1)
        return list(result.values)

    test = t.apply(lambda word: f(subtitle2,word.values),axis = 1)
    bow = dict( (k,v) for k,v in zip(vocab,test.values))

    #idf
    idf = {}
    N = len(list(bow.values())[0])

    for k,v in bow.items():
        if type(np.array(v) == 0) == bool:
            df = 1 if np.array(v) == 0 else 0
        else:
            df = sum(np.array(v) == 0)
        df = N - df 
        result = math.log(N / df)
        idf[k] = result

    #weight
    weight = [] 
    for num,v in enumerate(demo[col]):
        result = []
        for element in v:  
            result.append(bow[element][demo['v_id'].values[num]-start_v_id]*idf[element])

        weight.append(result)

    for i in range(len(weight)):
        w_sum = sum(weight[i])
        if w_sum == 0:
            w_sum = 1
        weight[i] = list(np.array(weight[i])/w_sum)
    return weight

#kor_demo['kor_w'] = get_w(kor_demo,kor_subtitle2,'morphs',0)
#eng_demo['eng_w'] = get_w(eng_demo,eng_subtitle2,'morphs',len(kor_subtitle2))

kor_demo['r_kor_w'] = get_w(kor_demo,kor_subtitle2,'text_token',0)
eng_demo['r_eng_w'] = get_w(eng_demo,eng_subtitle2,'text_token',len(kor_subtitle2))

print("complete.\n")

# 4. mean을 이용한 sentence to vector

def m_seq2vec(lst,model):
    size = len(lst)
    result = np.zeros(300,dtype = int)
    
    for i in range(size):
        try:
            dummy = model.wv[lst[i]]
            tmp = [ x + y for x,y in zip(result,dummy) ]
            result = tmp
        except:
            continue
            
    for i in range(300):
        try:
            result[i] = result[i]/size
        except:
            continue

    return result

## 5. tf_idf weight를 이용한 sentence to vector
def w_w2v_seq2vec(lst,model,w):
    size = len(lst)
    result = list(np.zeros(300,dtype = int))

    if len(w) != len(lst):
        return -1
    
    for i in range(size):
        try:
            dummy = np.array(model.wv[lst[i]])
            dummy = dummy*w[i]
            tmp = [ x + y for x,y in zip(result,dummy) ]
            result = tmp
        except:
            continue

    return result

def w_ft_seq2vec(lst,model,w):
    size = len(lst)
    result = list(np.zeros(300,dtype = int))

    if len(w) != len(lst):
        return -1
    
    for i in range(size):
        try:
            dummy = np.array(model[lst[i]])
            dummy = dummy*w[i]
            tmp = [ x + y for x,y in zip(result,dummy) ]
            result = tmp
        except:
            continue

    return result

if model == 'w2v':
    #kor_demo['w2v_vec'] = kor_demo.apply(lambda row: w_w2v_seq2vec(row['morphs'],kor_w2v_model,row['kor_w']),axis = 1)
    kor_demo['r_w2v_vec'] = kor_demo.apply(lambda row: w_w2v_seq2vec(row['text_token'],kor_w2v_model,row['r_kor_w']),axis = 1)
    #eng_demo['w2v_vec'] = eng_demo.apply(lambda row: w_w2v_seq2vec(row['morphs'],eng_w2v_model,row['eng_w']),axis = 1)
    eng_demo['r_w2v_vec'] = eng_demo.apply(lambda row: w_w2v_seq2vec(row['text_token'],eng_w2v_model,row['r_eng_w']),axis = 1)

elif model == 'ft':
    #kor_demo['ft_vec'] = kor_demo.apply(lambda row: w_ft_seq2vec(row['morphs'],d,row['kor_w']),axis = 1)
    kor_demo['r_ft_vec'] = kor_demo.apply(lambda row: w_ft_seq2vec(row['text_token'],d,row['r_kor_w']),axis = 1)
    #eng_demo['ft_vec'] = 0
    eng_demo['r_ft_vec'] = 0

else:
    #kor_demo['w2v_vec'] = kor_demo.apply(lambda row: w_w2v_seq2vec(row['morphs'],kor_w2v_model,row['kor_w']),axis = 1)
    kor_demo['r_w2v_vec'] = kor_demo.apply(lambda row: w_w2v_seq2vec(row['text_token'],kor_w2v_model,row['r_kor_w']),axis = 1)
    #eng_demo['w2v_vec'] = eng_demo.apply(lambda row: w_w2v_seq2vec(row['morphs'],eng_w2v_model,row['eng_w']),axis = 1)
    eng_demo['r_w2v_vec'] = eng_demo.apply(lambda row: w_w2v_seq2vec(row['text_token'],eng_w2v_model,row['r_eng_w']),axis = 1)

    #kor_demo['ft_vec'] = kor_demo.apply(lambda row: w_ft_seq2vec(row['morphs'],d,row['kor_w']),axis = 1)
    kor_demo['r_ft_vec'] = kor_demo.apply(lambda row: w_ft_seq2vec(row['text_token'],d,row['r_kor_w']),axis = 1)
    #eng_demo['ft_vec'] = 0
    eng_demo['r_ft_vec'] = 0

del kor_demo['kor_w']
del kor_demo['r_kor_w']
del eng_demo['eng_w']
del eng_demo['r_eng_w']

frames = [kor_demo,eng_demo]
final = pd.concat(frames)
final = final.reset_index()

final.to_csv(path + "/data/morphs_vec.csv")
