import pandas as pd
import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import math
import plotly.graph_objects as go
import pandas_datareader.data as get_data
import datetime

st.write('Final Project')

start_date = datetime.datetime(2017, 1, 1)
end_date = datetime.datetime(2021, 12, 31)

AAPL = get_data.get_data_yahoo("AAPL", start_date, end_date)
st.dataframe(AAPL)
