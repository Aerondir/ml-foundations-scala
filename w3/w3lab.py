
# coding: utf-8

# In[1]:

import graphlab


# # Read some product review data

# In[2]:

products = graphlab.SFrame('amazon_baby.gl/')


# # Lets explore the data

# In[3]:

products.head()


# # Build the word count vector for each review

# In[4]:

products['word_count'] = graphlab.text_analytics.count_words(products['review'])


# In[5]:

products.head()


# In[6]:

graphlab.canvas.set_target('ipynb')


# In[7]:

products['name'].show()


# # Explore Vulli Sophie

# In[10]:

giraffe_reviews = products[products['name'] == 'Vulli Sophie the Giraffe Teether']


# In[11]:

len(giraffe_reviews)


# In[13]:

giraffe_reviews['rating'].show(view='Categorical')


# # Build a sentiment classifier

# In[14]:

products['rating'].show(view='Categorical')


# ## Define what is a positive and a negative sentiment

# In[15]:

#ignore all 3 star reviews
products = products[products['rating']!=3]


# In[16]:

#positive sentiment is 4 or 5* reviews
products['sentiment'] = products['rating'] >=4


# In[18]:

products.head()


# ## Let's train the sentiment classifier

# In[19]:

train_data,test_data= products.random_split(.8, seed=0)


# In[21]:

sentiment_model = graphlab.logistic_classifier.create(train_data, 
                                                      target = 'sentiment', 
                                                      features=['word_count'], 
                                                      validation_set=test_data)


# # Evaluate the sentiment model

# In[22]:

sentiment_model.evaluate(test_data, metric='roc_curve')


# In[23]:

sentiment_model.show(view = 'Evaluation')


# # Applying the learned model to understand sentiment for Giraffe

# In[24]:

giraffe_reviews['predicted_sentiment'] = sentiment_model.predict(giraffe_reviews, output_type='probability')


# In[25]:

giraffe_reviews.head()


# ## Sort the reviews by the predicted sentiment and explore

# In[26]:

giraffe_reviews = giraffe_reviews.sort('predicted_sentiment', ascending=False)


# In[27]:

giraffe_reviews.head()


# In[28]:

giraffe_reviews[0]['review']


# In[29]:

giraffe_reviews[1]['review']


# # Show most negative reviews

# In[30]:

giraffe_reviews[-1]['review']


# In[31]:

giraffe_reviews[-2]['review']


# In[ ]:



