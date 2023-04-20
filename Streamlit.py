import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Configuration de Seaborn pour Streamlit
sns.set_theme(color_codes=True)

# Configuration de pandas pour afficher toutes les colonnes
st.set_option('deprecation.showPyplotGlobalUse', False)
pd.set_option('display.max_columns', None)

st.set_page_config(layout="wide")
# Afficher le titre en haut et centré
st.markdown("<h1 style='text-align: center;'>Data Slayers ⚔️</h1>", unsafe_allow_html=True)

# Importer les données
@st.cache
def load_data():
    df = pd.read_csv('Data.csv')
    return df

df = load_data()

# Supprimer les colonnes "Unnamed: 0" et "ID" 
df.drop(columns=['Unnamed: 0', 'id'], inplace=True)

# Analyse exploratoire des données

# Afficher le nombre de passagers satisfaits et insatisfaits
satisfaction_count = df.satisfaction.value_counts()
labels = ["Neutral or dissatisfied", "Satisfied"]
colors = sns.color_palette("YlOrBr")
fig1, ax1 = plt.subplots()
ax1.pie(satisfaction_count, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
ax1.axis('equal') 
ax1.set_title("Passenger satisfaction")

st.pyplot(fig1)

# Afficher les graphiques de comptage pour les variables catégorielles
cat_vars = ['Gender', 'Customer Type', 'Type of Travel', 'Class']

fig2, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
axs = axs.flatten()

for i, var in enumerate(cat_vars):
    sns.countplot(x=var, hue='satisfaction', data=df, ax=axs[i])
    axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=90)

fig2.tight_layout()

st.pyplot(fig2)

# Afficher les histogrammes pour les variables numériques
num_vars = ['Age', 'Flight Distance']

fig3, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
axs = axs.flatten()

for i, var in enumerate(num_vars):
    sns.histplot(x=var, data=df, ax=axs[i])

fig3.tight_layout()

st.pyplot(fig3)

# Afficher le graphique de dispersion
fig4 = plt.figure(figsize=(8,8),dpi=100)
sns.scatterplot(x="Departure Delay in Minutes", y="Arrival Delay in Minutes", hue="satisfaction", data=df, edgecolor="black")
st.pyplot(fig4)
