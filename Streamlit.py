import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_option('deprecation.showPyplotGlobalUse', False)
sns.set_theme(color_codes=True)
pd.set_option('display.max_columns', None)

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



Aghiles ABBACHE
je veu meme les shape etc
Voici le code mis à jour avec les informations de forme affichées :

scss
Copy code
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="Data Slayers ⚔️")

st.title("Exploration de données pour la satisfaction des passagers aériens")

df = pd.read_csv('Data.csv')

st.write("Shape of the dataset: ", df.shape)
st.write("_______________________________________")
st.write("Number of null values: ")
st.write(df.isnull().sum())
st.write("_______________________________________")
st.write("Number of unique values: ")
st.write(df.nunique())

st.write("## Supprimer les colonnes 'Unnamed: 0' et 'id'")

df.drop(columns=['Unnamed: 0', 'id'], inplace=True)

st.write("Nouvelle forme du jeu de données après suppression des colonnes : ", df.shape)

st.write("## Analyse exploratoire des données")

st.write("### Proportion de satisfaction")

satisfaction_counts = df.satisfaction.value_counts()
satisfaction_piechart = plt.pie(satisfaction_counts, labels = ["Neutral or dissatisfied", "Satisfied"], colors = sns.color_palette("YlOrBr"), autopct = '%1.1f%%')
st.pyplot(satisfaction_piechart.figure)

st.write("### Satisfaction en fonction des variables catégorielles")

cat_vars = ['Gender', 'Customer Type', 'Type of Travel', 'Class']

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
axs = axs.flatten()

for i, var in enumerate(cat_vars):
    sns.countplot(x=var, hue='satisfaction', data=df, ax=axs[i])
    axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=90)

fig.tight_layout()

st.pyplot(fig)

st.write("### Top 10 des tranches d'âge les plus voyageuses")

age_count = df['Age'].value_counts(ascending=False).head(10)
age_barplot = px.bar(y=age_count.values, 
             x=age_count.index, 
             color = age_count.index,
             color_discrete_sequence=px.colors.sequential.PuBuGn,
             text=age_count.values,
             title= "Top 10 des tranches d'âge les plus voyageuses :",
             template= 'plotly_dark')
age_barplot.update_layout(
    xaxis_title="Âge",
    yaxis_title="Nombre de voyages",
    font = dict(size=20,family="Franklin Gothic"))
st.plotly_chart(age_barplot)
