#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[1]:


import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns
import re
import nltk


# In[2]:


train = pd.read_csv("./fake_news_train.csv/train.csv")
test = pd.read_csv("./fake_news_test.csv/test.csv")


# In[3]:


print(train.shape, test.shape)


# In[4]:


train.head()


# In[5]:


print(train.isnull().sum())
print('************')
print(test.isnull().sum())


# In[6]:


test=test.fillna(' ')
train=train.fillna(' ')
test['total']=test['title']+' '+test['author']+test['text']
train['total']=train['title']+' '+train['author']+train['text']


# # Creating Wordcloud Visuals

# In[11]:


real_words = ''
fake_words = ''
stopwords = set(STOPWORDS) 
  
# iterate through the csv file 
for val in train[train['label']==1].total: 
  
    # split the value 
    tokens = val.split() 
      
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
      
    real_words += " ".join(tokens)+" "

for val in train[train['label']==0].total: 
      
    # split the value 
    tokens = val.split() 
      
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
      
    fake_words += " ".join(tokens)+" "


# In[12]:


wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(real_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 


# # Stopwords Generation

# In[7]:


from nltk.corpus import stopwords

stop_words = stopwords.words('english')
print(stop_words)


# # Lemmatization

# In[8]:


from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()


# # Regex(Cleaning), Tokenization, Stopward removal

# In[9]:


lemmatizer=WordNetLemmatizer()
for index,row in train.iterrows():
    filter_sentence = ''
    
    sentence = row['total']
    sentence = re.sub(r'[^\w\s]','',sentence) #cleaning
    
    words = nltk.word_tokenize(sentence) #tokenization
    
    words = [w for w in words if not w in stop_words]  #stopwords removal
    
    for word in words:
        filter_sentence = filter_sentence + ' ' + str(lemmatizer.lemmatize(word)).lower()
        
    train.loc[index,'total'] = filter_sentence


# In[10]:


train = train[['total','label']]
train.head()


# # NLP TECHNIQUES

# In[13]:


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[14]:


X_train = train['total']
Y_train = train['label']


# # TF - IDF and Count Vectorizer

# In[15]:


#Feature extraction using count vectorization and tfidf.
count_vectorizer = CountVectorizer()
count_vectorizer.fit_transform(X_train)
freq_term_matrix = count_vectorizer.transform(X_train)
tfidf = TfidfTransformer(norm="l2")
tfidf.fit(freq_term_matrix)
tf_idf_matrix = tfidf.fit_transform(freq_term_matrix)


# In[16]:


tf_idf_matrix


# # Modelling

# In[17]:


test_counts = count_vectorizer.transform(test['total'].values)
test_tfidf = tfidf.transform(test_counts)

#split in samples
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tf_idf_matrix, Y_train, random_state=0)


# # Logistic Regression

# In[18]:


from sklearn.linear_model import LogisticRegression


# In[19]:


logreg = LogisticRegression(C=1e5)
logreg.fit(X_train, y_train)
pred = logreg.predict(X_test)
print('Accuracy of Logistic Regression classifier on training set: {:.2f}'
     .format(logreg.score(X_train, y_train)))
print('Accuracy of Logistic Regression classifier on test set: {:.2f}'
     .format(logreg.score(X_test, y_test)))
from sklearn.naive_bayes import MultinomialNB
cm = confusion_matrix(y_test, pred)
cm


# # Multinomial Naive Bayes

# In[20]:


from sklearn.naive_bayes import MultinomialNB

NB = MultinomialNB()
NB.fit(X_train, y_train)
pred = NB.predict(X_test)
print('Accuracy of NB  classifier on training set: {:.2f}'
     .format(NB.score(X_train, y_train)))
print('Accuracy of NB classifier on test set: {:.2f}'
     .format(NB.score(X_test, y_test)))
cm = confusion_matrix(y_test, pred)
cm


# # Bernoulli Naive Bayes

# In[21]:


from sklearn.naive_bayes import BernoulliNB

bNB = BernoulliNB()
bNB.fit(X_train, y_train)
pred = bNB.predict(X_test)
print('Accuracy of bNB  classifier on training set: {:.2f}'
     .format(bNB.score(X_train, y_train)))
print('Accuracy of bNB classifier on test set: {:.2f}'
     .format(bNB.score(X_test, y_test)))
cm = confusion_matrix(y_test, pred)
cm


# # Prediction

# In[22]:


prediction = logreg.predict(X_test)
print(prediction)


# In[23]:


prediction.shape


# In[24]:


Id = test['id']


# In[25]:


dfs = pd.DataFrame(data=prediction,index=Id,columns=['label'])
dfs.head()


# In[26]:


dfs.to_csv(" FakeNewsPrediction.csv",index_label=['Id'],index=True)


# In[ ]:




