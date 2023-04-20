import pandas as pd
import time
import math
import streamlit as st
from io import StringIO
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

st.set_page_config(layout="wide")
st.title("Data Slayers ⚔️")

# Définir dataframe comme une variable globale
dataframe = None

# Onglets
tab1, tab2 = st.tabs(
    ["Upload & data exploration", "Prétraitement des données"])

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
            st.dataframe(dataframe.describe())
            st.write("Informations globales :")
            st.dataframe(dataframe.info)

    else: 
        st.warning("Veuillez choisir un fichier CSV")


with tab2:
    if dataframe is not None:
        st.title("Prétraitement des données")

        def preprocess_column(column, option, x=None):
            if option == "Pas de modification":
                pass
            if option == "Supprimer NaN":
                column = column.fillna(value=np.nan).dropna()
            elif option == "Fill par 0":
                column = column.fillna(0)
            elif option == "Fill Mean":
                column = column.fillna(column.mean())
            elif option == "Fill Median":
                column = column.fillna(column.median())
            elif option == "Encoding":
                encoder = LabelEncoder()
                column = encoder.fit_transform(column)
            elif option == "Arrondir":
                column = column.round(x)
            return column

        def split_list(lst):
            middle = math.ceil(len(lst) / 2)
            return lst[:middle], lst[middle:]

        col1, col2 = st.columns(2)
        col_names1, col_names2 = split_list(dataframe.columns)
        col1, col2 = st.columns(2)
        dataframe_0 = dataframe.copy()
        x = st.number_input(
            "Choisir le nombre de décimales",
            min_value=1,
            max_value=4,
            value=1)
        with col1:
            for col in col_names1:
                option = st.radio(
                    f"Choisir une option de prétraitement pour la colonne {col}",
                    [
                        "Pas de modification",
                        "Supprimer NaN",
                        "Fill par 0",
                        "Fill Mean",
                        "Fill Median",
                        "Encoding",
                        "Arrondir"])
                processed_col = preprocess_column(dataframe[col], option, x)
                dataframe_0[col] = processed_col
        with col2:
            for col in col_names2:
                option = st.radio(
                    f"Choisir une option de prétraitement pour la colonne {col}",
                    [
                        "Pas de modification",
                        "Supprimer NaN",
                        "Fill par 0",
                        "Fill Mean",
                        "Fill Median",
                        "Encoding",
                        "Arrondir"])
                processed_col = preprocess_column(dataframe[col], option, x)
                dataframe_0[col] = processed_col
            old_val = None
            new_val = None
            replace_val = st.text_input("Remplacer les valeurs", value=old_val)
            by_val = st.text_input("Remplacer par", value=new_val)
            if replace_val != old_val or by_val != new_val:
                dataframe_0_new = dataframe_0.replace(replace_val, by_val)
                dataframe_0 = dataframe_0_new
                old_val = replace_val
                new_val = by_val

        col3, col4 = st.columns(2)
        with col3:
            st.write("Données avant prétraitement")
            st.dataframe(dataframe)
            st.write(dataframe.shape)
        with col4:
            st.write("Données après prétraitement")
            st.dataframe(dataframe_0)
            st.write(dataframe_0.shape)

        @st.cache_data
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every
            # rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(dataframe_0)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='output.csv',
            mime='text/csv',
        )
