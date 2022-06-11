import pandas as pd
import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import math
import plotly.graph_objects as go
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By

st.write('Final Project')

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get("https://lenta.com/allmarkets/")

elements = driver.find_elements(by=By.TAG_NAME, value="li")
for e in elements:
    st.write(e.text)
