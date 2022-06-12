import pandas as pd
import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import math
import plotly.graph_objects as go
import sqlite3

st.write('Final Project')

conn = sqlite3.connect("database.sqlite")
c = conn.cursor()
c.execute(
    """
CREATE TABLE gradebook (
id integer PRIMARY KEY,
first_name text,
last_name text,
grade integer
)
"""
)

c.execute(
    """
SELECT * FROM gradebook;
"""
).fetchall()
