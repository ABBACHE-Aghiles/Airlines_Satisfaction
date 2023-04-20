import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import StringIO
sns.set_theme(color_codes=True)
pd.set_option('display.max_columns', None)
import streamlit as st

st.set_page_config(layout="wide")
st.title("Data Slayers ⚔️")

# Définir dataframe comme une variable globale
dataframe = None

# Onglets
tab1 , tab2 = st.tabs(
    ["Upload & data exploration", "Visualisation"])


# Page d'analyse
with tab1:
    uploaded_file = st.file_uploader("Choisissez un fichier CSV")
    gauche, droite = st.columns(2)
    if uploaded_file is not None:
        if uploaded_file.type == 'text/csv':

            with gauche:
                sep = st.selectbox(
                    "Sélectionner un séparateur :", [
                        ",", ";", "tab"])
            with droite:
                encoding = st.selectbox(
                    "Sélectionner un formatage :", [
                        "UTF-8", "ISO-8859-1", "ASCII", "UTF_8_SIG", "UTF_16", "CP437"])
            if sep == sep:
                dataframe = pd.read_csv(
                    StringIO(
                        uploaded_file.getvalue().decode(encoding)),
                    sep=sep)
                st.dataframe(dataframe)

            else:
                dataframe = pd.read_csv(
                    StringIO(
                        uploaded_file.getvalue().decode(encoding)),
                    sep="/t")
                st.dataframe(dataframe)
            with droite:
                float_cols = dataframe.select_dtypes(
                    include=['float64']).columns
                dataframe[float_cols] = dataframe[float_cols].astype(
                    'float32')
                st.write("Type de données :")
                st.dataframe(dataframe.dtypes)
            with gauche:
                st.write("Nombre de valeurs manquantes :")
                st.dataframe(dataframe.isnull().sum())
            st.write("Nombre de lignes et colonnes", dataframe.shape)
            st.write("Statistiques descriptives :")
            st.dataframe(dataframe.describe().T.style.set_properties(**{"background-color": "#FBA7A7", "font-size" : "17px",
                                        "color": "#ffffff", "border-radius" : "1px", "border": "1.5px solid black"}))
            st.write("Nombre de valeurs uniques :")
            st.dataframe(dataframe.nunique())
            st.write("Vous pouvez constater ce qui suit :")
            st.write("1) La colonne correspondant à la caractéristique Délai d'arrivée en minutes comporte 310 valeurs manquantes.")
            st.write("2) Les deux premières caractéristiques sont inutiles et n'affecteront pas la classification, nous devrions donc nous en débarrasser.")
            
            

    else: 
        st.warning("Veuillez choisir un fichier CSV")

