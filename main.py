import pandas as pd
import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import pandas_datareader.data as get_data
import datetime
from sklearn.linear_model import LinearRegression

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

start_date = datetime.datetime(2017, 1, 1)
end_date = datetime.datetime(2021, 12, 31)

AAPL = get_data.get_data_yahoo("AAPL", start_date, end_date)

df = pd.read_csv('Stores.csv')
st.markdown('''This is the "Supermarket store branches sales analysis" dataset''')
st.dataframe(df)
st.markdown('''Here is the correlation matrix for all the attributes in the dataset''')
st.dataframe(df.corr())
st.markdown('''This graph shows distributions of store parameters''')
fig = ff.create_distplot([df['Store_Area'], df['Items_Available'], df['Daily_Customer_Count']], group_labels=['Area', 'Items', 'Customers'])
st.plotly_chart(fig, use_container_width=True)
st.markdown('''Here are calculates mean and variance of the parameters''')
st.dataframe(data=df.drop(columns=['Store_Sales']).mean().rename(columns={0:"Mean"}))

st.subheader('Sales prediction')

st.markdown('''In this section you are prompted to choose hypothetical expected values for the corresponding 
features of the store. In this section a multivariate linear regression of the form:''')
st.latex(r'''
\text{Store Sales} = X_0 + X_1\times\text{area} + X_2\times\text{items} + X_3\times\text{customers}
''')

multivar_model = LinearRegression()
multivar_model.fit(df.drop(columns=['Store ID ', 'Store_Sales']), df['Store_Sales'])

area = st.slider("Choose expected store area", min_value = int(df['Store_Area'].min()), max_value = int(df['Store_Area'].max()), step=5)
items_num = st.slider("Choose expected item numer", min_value = int(df['Items_Available'].min()), max_value = int(df['Items_Available'].max()), step=5)
customers_num = st.slider("Choose expected daily customer count", min_value = int(df['Daily_Customer_Count'].min()),
                          max_value = int(df['Daily_Customer_Count'].max()), step=5)

predicted_sales = multivar_model.predict([[area, items_num, customers_num]])
st.write(f'''Your sales will be: {predicted_sales}''')
