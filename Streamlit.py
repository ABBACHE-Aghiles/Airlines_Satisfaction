import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="Data Slayers ⚔️")
st.markdown("<h1 style='text-align: center;'>Data Slayers ⚔️</h1>", unsafe_allow_html=True)
st.title("Exploration de données pour la satisfaction des passagers aériens")

df = pd.read_csv('Data.csv')
st.write(df.head())
st.write(df.tail())
st.write("Shape of the dataset: ", df.shape)
st.write("_______________________________________")
st.write("Number of null values: ")
st.write(df.isnull().sum())
st.write("_______________________________________")
st.write("Number of unique values: ")
st.write(df.nunique())
