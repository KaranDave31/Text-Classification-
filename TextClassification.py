#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


yelp = pd.read_csv('yelp.csv')


# In[3]:


yelp.head()


# In[4]:


yelp.info()


# In[5]:


yelp.describe()


# In[6]:


yelp['text_length'] = yelp['text'].apply(len)


# In[7]:


sns.set_style('white')


# In[12]:


g = sns.FacetGrid(yelp,col='stars')
g.map(plt.hist,'text_length',bins=50)


# In[14]:


sns.boxplot(x='stars',y='text_length', data =yelp)


# In[15]:


sns.countplot(x='stars',data=yelp)


# In[19]:


stars = yelp.groupby('stars').mean(numeric_only=True)
stars


# In[20]:


stars.corr()


# In[23]:


sns.heatmap(stars.corr(),cmap='coolwarm',annot=True)


# In[24]:


yelp_class = yelp[(yelp['stars'] == 1) | (yelp['stars'] == 5)]


# In[25]:


yelp_class.info()


# In[26]:


X = yelp_class['text']
y = yelp_class['stars']


# In[27]:


from sklearn.feature_extraction.text import CountVectorizer


# In[28]:


cv = CountVectorizer()


# In[29]:


X = cv.fit_transform(X)


# In[30]:


from sklearn.model_selection import train_test_split


# In[36]:


X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3,random_state=101)


# In[37]:


from sklearn.naive_bayes import MultinomialNB


# In[38]:


nb = MultinomialNB()


# In[39]:


nb.fit(X_train,y_train)


# In[40]:


predictions = nb.predict(X_test)


# In[41]:


from sklearn.metrics import confusion_matrix,classification_report


# In[42]:


print(confusion_matrix(y_test,predictions))


# In[43]:


print(classification_report(y_test,predictions))


# In[45]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[46]:


from sklearn.pipeline import Pipeline


# In[47]:


pipe = Pipeline([('bow',CountVectorizer()),
                 ('tfidf',TfidfTransformer()),
                 ('model',MultinomialNB())])


# In[48]:


X = yelp_class['text']
y = yelp_class['stars']

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3,random_state=101)


# In[49]:


pipe.fit(X_train,y_train)


# In[50]:


predictions = pipe.predict(X_test)


# In[51]:


print(confusion_matrix(y_test,predictions))


# In[52]:


print(classification_report(y_test,predictions))


# In[ ]:




