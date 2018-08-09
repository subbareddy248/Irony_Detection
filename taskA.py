# -*- coding: utf-8 -*-
from __future__ import print_function
import libspacy
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import sys
from sklearn.utils import shuffle
from nltk.corpus import sentiwordnet as swn
import preprocessor as p
import pickle

import mlp
import logreg
import xgb
import rf
import svm
import tf_idf
import libsenti
import auto_feat

reload(sys)
sys.setdefaultencoding('utf8')
auto_features=[]
def main():
  global auto_features

  train_file = 'SemEval2018-T4-train-taskA.txt'
  test_file = 'SemEval2018-T3_input_test_taskA.txt'
  print("Beginning Irony-TaskA")
  p.set_options(p.OPT.URL, p.OPT.EMOJI)
  #Read the train_file
  raw_train=[]
  df = pd.read_csv(train_file, delimiter='\t', header=None, skiprows=1)
  for (tid, label, text) in zip(df[0], df[1], df[2]):
    text = unicode(text, errors='ignore')
    text = p.clean(text)
    raw_train.append([tid, text, int(label)])
    #print(tid, text, "label=", label)

  print("Loaded %d training samples" % len(raw_train))

  raw_test=[]
  df = pd.read_csv(test_file, delimiter='\t', header=None, skiprows=1)
  for (tid, text) in zip(df[0], df[1]):
    raw_test.append([tid, text])

  print("Loaded %d testing samples" % len(raw_test))

  X_all=[]
  y_all=[]
  ids_all=[]
  c1=[]
  c0=[]
  for sample in raw_train:
    tweet = process_tweet(sample[1])
    y=sample[2] #real_label
    if y==0:
      c0.append(sample[1])
    else:
      c1.append(sample[1])
  write_to_file('c0.txt', c0)
  write_to_file('c1.txt', c1)
  feats = auto_feat.identify_features(c1,c0,400)
  auto_features=[i[0] for i in feats]
  print(auto_features)
  #Generate features
  for sample in raw_train:
    tweet = process_tweet(sample[1])
    X = generate_features(tweet)
    y=sample[2] #real_label
    X_all.append(X)
    y_all.append(y)
    ids_all.append(sample[0])
  print("Writing to class specific files")

  X_real_test=[] 
  ids_real_test=[]
  for sample in raw_test:
    tweet = process_tweet(sample[1])
    X = generate_features(tweet)
    X_real_test.append(X)
    #ids_real_test.append(tid)
  #X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.20, random_state=42)
  #X_tf_train, X_tf_test, y_tf_train, y_tf_test = train_test_split([s[1] for s in raw_train], y_all, test_size=0.20, random_state=42)
  #tf_idf.fit_predict(X_tf_train, y_tf_train, X_tf_test, y_tf_test)

  print("Training with ", len(X_all))
  print("Testing with ", len(X_real_test))
  print("#Tr.Features=", len(X_all[0]))
  print("#Te.Features=", len(X_real_test[0]))
  print("#Tr.positive=", sum(y_all))
  print("#Te.positive=", sum(y_test))

  
  funcs= [
            (logreg.fit_predict,"Logistic Regression"),
            (rf.fit_predict,"Random Forest"),
            (svm.fit_predict,"Support Vector Machine"),
            (xgb.fit_predict, "XGBoost"),
           ]
  #funcs=[]
  for fit_predict,name in funcs:
    print("Running function", name)
    fit_predict(X_all, y_all, X_real_test, y_test=[i for i in X_real_test])

  model = mlp.fit_predict(X_all, y_all, X_real_test, y_test=[i for i in X_real_test])


  #load the picke file
  #pickle_file = 'xgb.pickle'
  #model = pickle.load( open( pickle_file, "rb" ) )
  y_pred = model.predict(X_real_test)
  fp=open('mlp-predictions-taskA.txt','w')
  for y in y_pred:
    fp.write((str(y)+'\n'))
  fp.close()
  '''
  print("Predicting on real test")
  y_real_probas = logistic.predict_proba(X_real_test).tolist()

 
  with open('results.txt','w') as fp:
    fp.write('"id","EAP","HPL","MWS"'+'\n')
    for tid, scores in zip(ids_real_test, y_real_probas):
      fp.write('"'+tid+'"'+','+str(scores[0])+','+str(scores[1])+','+str(scores[2])+'\n')
  '''

def write_to_file(filename, lines):
  with open(filename,'w') as fp:
    for line in lines:
      fp.write(line+'\n')
anger_words=['anger','fuming','rag','loath','bitter','furi','offend','hate','piss','sad','emo','fuck']
def generate_features(text):
  features=[]
  tokens = text.split()
  X = libspacy.get_vector(text)
  return X.tolist()
 
  f=len(text)  #Length of the text
  features.append(f)  

  f=text.count('.')
  features.append(f)

  f=text.count('#')
  features.append(f)

  f = text.count('I ')
  #features.append(f)


  f=text.count('@')
  features.append(f)
    
  f=text.count('?')
  features.append(f)

  f=text.count('irony')
  features.append(f)

  oh=[text.count(w) for w in auto_features]
  #print(sum(oh))
  features +=oh
  senti = sum(libsenti.get_sentiments(text))
  #print(text)
  #features.append(senti)

    
  return features
  return features+X.tolist()
 
def replace_trash(unicode_string):
  for i in range(0, len(unicode_string)):
    try:
      unicode_string[i].encode("ascii")
    except:
      #means it's non-ASCII
     unicode_string=unicode_string[i].replace(" ") #replacing it with a single space
  return unicode_string

def process_tweet(tweet):
  #tweet = tweet.replace('#',' ')
  return tweet
if __name__=="__main__":
  main()
