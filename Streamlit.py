import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import StringIO
sns.set_theme(color_codes=True)
pd.set_option('display.max_columns', None)
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
            # Supprimer les colonnes "Unnamed: 0" et "ID"
            dataframe.drop(columns=['Unnamed: 0', 'id'], inplace=True)

            # Afficher le code pour supprimer les colonnes
            st.write("# Supprimer les colonnes \"Unnamed: 0\" et \"ID\"")
            st.write("dataframe.drop(columns=['Unnamed: 0', 'ID'], inplace=True)")

            # Afficher la forme du DataFrame mis à jour
            st.write("La forme du DataFrame mis à jour est : ", dataframe.shape)
        
            # Afficher les informations du DataFrame mis à jour
            st.write("LeDataFrame mis à jour : ")
            st.write(dataframe.head())



            

    else: 
        st.warning("Veuillez choisir un fichier CSV")
        
        
with tab2:
    if dataframe is not None:
        st.write("# Analyse exploratoire des données")
        # Créer un graphique camembert pour la colonne "satisfaction"
        fig, ax = plt.subplots()
        ax.pie(dataframe.satisfaction.value_counts(), labels=["Neutral or dissatisfied", "Satisfied"],
        colors=sns.color_palette("YlOrBr"), autopct='%1.1f%%')
        ax.set_title("Satisfaction")
        # Afficher le graphique camembert
        st.write(fig)
        st.write("Comme la pie chart le montre , la sélection est plus ou moins équilibrée.")
        # Liste des variables catégorielles à tracer
        cat_vars = ['Gender', 'Customer Type', 'Type of Travel', 'Class']

        # Créer une figure avec des sous-intrigues
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
        axs = axs.flatten()

        # Créer un diagramme à barres pour chaque variable catégorielle avec une boucle for
        for i, var in enumerate(cat_vars):
            
            sns.countplot(x=var, hue='satisfaction', data=dataframe, ax=axs[i])
            axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=90)

        # Ajuster l'espacement entre les sous-plots
        fig.tight_layout()

        # Afficher le graphique dans Streamlit
        st.pyplot(fig)
        # Calculer le décompte des âges
        age_count = dataframe['Age'].value_counts(ascending=False).head(10)

        # Créer un graphique à barres horizontal avec Plotly Express
        fig = px.bar(y=age_count.values, 
             x=age_count.index, 
             color = age_count.index,
             color_discrete_sequence=px.colors.sequential.PuBuGn,
             text=age_count.values,
             title= "Top 10 des tranches d'âge les plus voyageuses :",
             template= 'plotly_dark')

        # Mettre à jour les titres des axes et la taille de la police
        fig.update_layout(
                xaxis_title="Âge",
                yaxis_title="Nombre de voyages",
                font = dict(size=20,family="Franklin Gothic"))

        # Afficher le graphique dans Streamlit
        st.plotly_chart(fig)







        

        
