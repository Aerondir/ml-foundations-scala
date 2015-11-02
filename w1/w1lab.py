
# coding: utf-8

# # Fire up graphlab create

# In[1]:

import graphlab


# # Load some house sales data

# In[2]:

sales = graphlab.SFrame('home_data.gl/')


# In[3]:

sales


# # Exploring the data for housing sales

# In[4]:

graphlab.canvas.set_target('ipynb')
sales.show(view="Scatter Plot", x="sqft_living", y="price")


# # Create a simple regression model of sqft_living to price

# In[5]:

train_data,test_data = sales.random_split(.8,seed=0)


# # Build the regression model

# In[6]:

sqft_model = graphlab.linear_regression.create(train_data, target = 'price', features=['sqft_living'])


# # Evaluate the simple model

# In[7]:

print test_data['price'].mean()


# In[8]:

print sqft_model.evaluate(test_data)


# # Let's show what aour predictions look like

# In[11]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[13]:

plt.plot(test_data['sqft_living'], test_data['price'],'.',
         test_data['sqft_living'],sqft_model.predict(test_data),'-')


# In[15]:

sqft_model.get('coefficients')


# # Explore other features in the data

# In[17]:

my_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']


# In[18]:

sales[my_features].show()


# In[19]:

sales.show(view='BoxWhisker Plot', x='zipcode', y='price')


# # Build a regression model with more features

# In[61]:

my_features_model = graphlab.linear_regression.create(train_data, target='price', features = my_features, validation_set=None)


# In[21]:

print my_features


# In[23]:

print sqft_model.evaluate(test_data)
print my_features_model.evaluate(test_data)


# # Apply learned models to predict prices of 3 houses

# In[24]:

house1 = sales[sales['id']=='5309101200']


# In[25]:

house1


# <img src="house-5309101200.jpg">

# In[26]:

print house1['price']


# In[27]:

print sqft_model.predict(house1)


# In[28]:

print my_features_model.predict(house1)


# ## Prediction for a second, fancier house

# In[31]:

house2 = sales[sales['id']=='1925069082']


# In[32]:

house2


# In[33]:

sqft_model.predict(house2)


# In[34]:

print my_features_model.predict(house2)


# ## Last house, super fancy

# In[35]:

bill_gates = { 'bedrooms':[8], 
              'bathrooms':[25], 
              'sqft_living':[50000], 
              'sqft_lot':[225000],
              'floors':[4], 
              'zipcode':['98039'], 
              'condition':[10], 
              'grade':[10],
              'waterfront':[1],
              'view':[4],
              'sqft_above':[37500],
              'sqft_basement':[12500],
              'yr_built':[1994],
              'yr_renovated':[2010],
              'lat':[47.627606],
              'long':[-122.242054],
              'sqft_living15':[5000],
              'sqft_lot15':[40000]}


# In[36]:

print my_features_model.predict(graphlab.SFrame(bill_gates))


# In[37]:

zipcode = sales[sales['zipcode']=='98039']


# In[38]:

zipcode


# In[39]:

zipcode['price'].mean()


# In[41]:

logfil = sales[(sales['sqft_living']>=2000) & (sales['sqft_living']<=4000)]


# In[43]:

logfil


# In[46]:

lflen = len(logfil)


# In[47]:

alllen = len(sales)


# In[51]:

print lflen
print alllen


# In[53]:

9221/21613.0


# In[55]:

advanced_features = [
'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode',
'condition', # condition of house				
'grade', # measure of quality of construction				
'waterfront', # waterfront property				
'view', # type of view				
'sqft_above', # square feet above ground				
'sqft_basement', # square feet in basement				
'yr_built', # the year built				
'yr_renovated', # the year renovated				
'lat', 'long', # the lat-long of the parcel				
'sqft_living15', # average sq.ft. of 15 nearest neighbors 				
'sqft_lot15', # average lot size of 15 nearest neighbors 
]


# In[59]:

af_model = graphlab.linear_regression.create(train_data, target='price', features = advanced_features, validation_set=None)


# In[62]:

print my_features_model.evaluate(test_data)
print af_model.evaluate(test_data)


# In[63]:

179542.4333126908 - 156831.11680200786


# In[ ]:



