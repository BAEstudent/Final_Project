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

st.header('Here is the correlation matrix for the parameters in the dataset')

start_date = datetime.datetime(2017, 1, 1)
end_date = datetime.datetime(2021, 12, 31)

AAPL = get_data.get_data_yahoo("AAPL", start_date, end_date)

df = pd.read_csv('Stores.csv')
st.dataframe(df.corr())
fig = ff.create_distplot([df['Store_Area'], df['Daily_Customer_Count']], group_labels=['Area', 'Customers'])
st.plotly_chart(fig, use_container_width=True)

multivar_model = LinearRegression()
multivar_model.fit(df.drop(columns=['Store ID ', 'Store_Sales']), df['Store_Sales'])

area = st.slider("Choose expected store area", min_value = int(df['Store_Area'].min()), max_value = int(df['Store_Area'].max()), step=5)
items_num = st.slider("Choose expected item numer", min_value = int(df['Items_Available'].min()), max_value = int(df['Items_Available'].max()), step=5)
customers_num = st.slider("Choose expected daily customer count", min_value = int(df['Daily_Customer_Count'].min()),
                          max_value = int(df['Daily_Customer_Count'].max()), step=5)

predicted_sales = multivar_model.predict([[area, items_num, customers_num]])
st.write(predicted_sales)
