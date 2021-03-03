#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
import tensorflow as tf
import nltk
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import Dense,LSTM,Dropout,Embedding
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score


# In[38]:


df = pd.read_csv('Downloads/train.csv')


# In[39]:


df = df.dropna()


# In[40]:


X = df.drop('label',axis = 1)
y = df['label']
X


# In[41]:


voc_size = 5000
messages = X.copy()
messages.reset_index(inplace=True)


# In[42]:


nltk.download('stopwords')


# In[43]:


ps = PorterStemmer()
all_words =[]
for i in range(0, len(messages)):
    review =re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    all_words.append(review)


# In[44]:


onehot_repr = [one_hot(words,voc_size)for words in all_words]


# In[45]:


sent_length = 20
embedded_docs = pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
emb_vec_features = 40
def create_model():
    model = Sequential()
    model.add(Embedding(voc_size,emb_vec_features,input_length = sent_length))
    model.add(Dropout(0.3))
    model.add(LSTM(100))
    model.add(Dropout(0.3))
    model.add(Dense(1,activation = 'sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer ='adam',metrics=['accuracy'])
    return model


# In[46]:


X_f = np.array(embedded_docs)
y_f = np.array(y)
X_train, X_test, y_train,y_test=train_test_split(X_f,y_f,test_size=0.33, random_state=42)


# In[48]:


from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


# In[57]:


param_grid = [{'batch_size':[10,20,40,60,80,100],'epochs':[5,10,20]}]
# model.fit(X_train,y_train,validation_data=(X_test,y_test))
model_1 = KerasClassifier(build_fn=create_model, epochs = 10,batch_size = 64)
grid = GridSearchCV(model_1,param_grid,cv = 3,n_jobs = -1,scoring = 'neg_mean_squared_error')
grid_result = grid.fit(X_train,y_train)


# In[58]:


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[59]:


model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs = 5,batch_size = 100)


# In[60]:


y_pred = model.predict_classes(X_test)
confusion_matrix(y_test,y_pred),accuracy_score(y_test,y_pred)


# In[ ]:




