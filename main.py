import pandas as pd
import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import pandas_datareader.data as get_data
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import scipy as sp

st.title('Final Project')

st.subheader('Project discription')
st.markdown('''Hello! This streamlit app is a part of the final project for the course
in Data Science, which is dedicated to analyzing a dataset that contains information 
about supermarket store branches (area, items available, etc.) The dataset had been taken 
from kaggle (https://www.kaggle.com/datasets/surajjha101/stores-area-and-sales-data).
The ultimate goal of this app is to predict sales of such stores using machine learning. 
This app also provides some data analysis and visualization.

Here is a link to project's git repository: https://github.com/BAEstudent/Final_Project''')

st.subheader('Data dyscription and analysis')

df = pd.read_csv('Stores.csv')
st.markdown('''This is the "Supermarket store branches sales analysis" dataset''') 
st.dataframe(df)
st.markdown('''Here is the correlation matrix for all the attributes in the dataset''')
st.dataframe(df.corr())
st.markdown('''This graph shows distributions of store parameters''')
fig = ff.create_distplot([df['Store_Area'], df['Items_Available'], df['Daily_Customer_Count']], group_labels=['Area', 'Items', 'Customers']) ### Creating plot
st.plotly_chart(fig, use_container_width=True)
st.markdown('''Here are calculated mean and variance of the parameters''')
st.dataframe(                                                                                              ### Creating the mean and variance table
  data=df.drop(columns=['Store ID ', 'Store_Sales']).mean().reset_index().rename(columns={0:"Mean"}).join(
  df.drop(columns=['Store ID ', 'Store_Sales']).var().reset_index().rename(columns={0:"Variance"}), rsuffix='index'
  ).drop(columns=['indexindex'])
)

st.subheader('Sales prediction')

st.markdown('''In this section you are prompted to choose hypothetical expected values for the corresponding 
features of the store. In this section a multivariate linear regression of the form:''')
st.latex(r'''
\text{Store Sales} = X_0 + X_1\times\text{area} + X_2\times\text{items} + X_3\times\text{customers}
''')
st.markdown('''and a the KNN model are used.''')

st.markdown('''Here is the code chunk with models creation and training:''')
with st.echo():
    train_df, test_df = train_test_split(df) ### Creating training and testing samples
    multivar_model = LinearRegression()      ### Making the linear regression
    multivar_model.fit(train_df.drop(columns=['Store ID ', 'Store_Sales']), train_df['Store_Sales'])
    knn_model = KNeighborsRegressor()
    knn_model.fit(train_df.drop(columns=['Store ID ', 'Store_Sales']), train_df['Store_Sales'])
    
area = st.slider("Choose expected store area", min_value = int(df['Store_Area'].min()), max_value = int(df['Store_Area'].max()), step=5)
items_num = st.slider("Choose expected item numer", min_value = int(df['Items_Available'].min()), max_value = int(df['Items_Available'].max()), step=5)
customers_num = st.slider("Choose expected daily customer count", min_value = int(df['Daily_Customer_Count'].min()),
                          max_value = int(df['Daily_Customer_Count'].max()), step=5)

predicted_sales_mult = multivar_model.predict([[area, items_num, customers_num]])
st.write(f'''First, here is the result for the multivariate linear regression. Your sales will be: {predicted_sales_mult}''')


predicted_sales_knn = knn_model.predict([[area, items_num, customers_num]])
st.write(f'''First, here is the result for the KNN model. Your sales will be: {predicted_sales_knn}''')

st.markdown('''It is up to you to decide, which model to trust (I, personally, wouldn't trust any of these), so here are some model validity characteristics.''')

st.subheader('Multivariate Linear regression prediction graphs')

st.markdown('''The prediction was made based on a test sample from the original dataset. The models were trained
with the training sample from the dataset''')

fig_1, (ax_1_area, ax_1_item, ax_1_cust) = plt.subplots(3, 1, figsize=(5, 15))
ax_1_area.scatter(df['Store_Area'], df['Store_Sales'], c='tab:blue', label='Real data')
ax_1_area.scatter(test_df['Store_Area'],multivar_model.predict(test_df.drop(columns=['Store ID ', 'Store_Sales'])), c='tab:orange', label='Prediction')
ax_1_area.set_title('Store Area')
ax_1_area.set_xlabel('Store Area')
ax_1_area.set_ylabel('Store Sales')
ax_1_area.legend()
ax_1_item.scatter(df['Items_Available'], df['Store_Sales'], c='tab:blue', label='Real data')
ax_1_item.scatter(test_df['Items_Available'],multivar_model.predict(test_df.drop(columns=['Store ID ', 'Store_Sales'])), c='tab:orange', label='Prediction')
ax_1_item.set_title('Items Available')
ax_1_item.set_xlabel('Number of Items')
ax_1_item.set_ylabel('Store Sales')
ax_1_item.legend()
ax_1_cust.scatter(df['Daily_Customer_Count'], df['Store_Sales'], c='tab:blue', label='Real data')
ax_1_cust.scatter(test_df['Daily_Customer_Count'],multivar_model.predict(test_df.drop(columns=['Store ID ', 'Store_Sales'])), c='tab:orange', label='Prediction')
ax_1_cust.set_title('Daily Customer Count')
ax_1_cust.set_xlabel('Number of Customers')
ax_1_cust.set_ylabel('Store Sales')
ax_1_cust.legend()
st.pyplot(fig_1)

st.subheader('KNN model prediction graphs')
fig_2, (ax_2_area, ax_2_item, ax_2_cust) = plt.subplots(3, 1, figsize=(5, 15))
ax_2_area.scatter(df['Store_Area'], df['Store_Sales'], c='tab:blue', label='Real data')
ax_2_area.scatter(test_df['Store_Area'],knn_model.predict(test_df.drop(columns=['Store ID ', 'Store_Sales'])), c='tab:orange', label='Prediction')
ax_2_area.set_title('Store Area')
ax_2_area.set_xlabel('Store Area')
ax_2_area.set_ylabel('Store Sales')
ax_2_area.legend()
ax_2_item.scatter(df['Items_Available'], df['Store_Sales'], c='tab:blue', label='Real data')
ax_2_item.scatter(test_df['Items_Available'],knn_model.predict(test_df.drop(columns=['Store ID ', 'Store_Sales'])), c='tab:orange', label='Prediction')
ax_2_item.set_title('Items Available')
ax_2_item.set_xlabel('Number of Items')
ax_2_item.set_ylabel('Store Sales')
ax_2_item.legend()
ax_2_cust.scatter(df['Daily_Customer_Count'], df['Store_Sales'], c='tab:blue', label='Real data')
ax_2_cust.scatter(test_df['Daily_Customer_Count'],knn_model.predict(test_df.drop(columns=['Store ID ', 'Store_Sales'])), c='tab:orange', label='Prediction')
ax_2_cust.set_title('Daily Customer Count')
ax_2_cust.set_xlabel('Number of Customers')
ax_2_cust.set_ylabel('Store Sales')
ax_2_cust.legend()
st.pyplot(fig_2)

st.subheader('''A bit different topic... here is some math''')
st.markdown('''After finishing the machine learning part of the project I was thinking of ideas for implementation of math in python in my project.
At some point I remembered that in our macroeconomics-2 course we've talked about seasonal smoothing of time series and a thing called Hodrick-Prescott (HP) Filter.
And I had realised that we didn't have an opporunity to see it working in our macro course. So, I decided to implement here. I didn't find any better data
than Apple stock prices for the last five years, so I am really unsure of the value of such smoothed data... but the math is still valid.
Anyway, it was fun to try using this filter.

For this problem I was using scipy ans numpy modules. To save your time, here is the minimization problem of the filter that yields a vector
of smoothed data:''')
st.latex(r'''\displaystyle\min_{g_t} \ \left[\displaystyle\sum_{t=2}^T (y_t-g_t)^2 + 
\lambda\displaystyle\sum_{t=2}^{T}[(g_{t-1}-g_t)-(g_{t-1}-g_{t-2})^2]\right],''')
st.latex(r'''\text{where y - is the vector of original data, g - the vector of smoothed data (minimization variable)}\\, \lambda -
\text{a parameter, that was set to 1600.}''')



