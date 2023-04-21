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
tab1 , tab2, tab3, tab4, tab5,tab6 = st.tabs(
    ["Upload and explore DATA ♥", "Visualisation","Random Forest","K-Means", "Régression Logistique","DNN"])


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
            # Vérification des valeurs nulles
            check_missing = dataframe.isnull().sum() * 100 / dataframe.shape[0]
            missing_cols = check_missing[check_missing > 0].sort_values(ascending=False).index.tolist()

            if missing_cols:
                st.warning(f"Il y a des valeurs manquantes dans les colonnes suivantes : {', '.join(missing_cols)}")
                dataframe.fillna(dataframe.median(), inplace=True)
            else:
                st.success("Aucune valeur manquante dans le jeu de données")

            # Affichage des types de données
            st.write("Types de données :")
            st.write(dataframe.dtypes)

            # Affichage des valeurs uniques pour les colonnes catégorielles
            st.write("Valeurs uniques pour les colonnes catégorielles :")
            for col in dataframe.select_dtypes(include=['object']).columns:
                st.write(f"{col}: {dataframe[col].unique()}")
                
            # Vérifier les valeurs nulles
            check_missing = dataframe.isnull().sum() * 100 / dataframe.shape[0]

            # Afficher les valeurs manquantes
            if check_missing.any():
                st.write('Les colonnes avec des valeurs manquantes :')
                st.write(check_missing[check_missing > 0].sort_values(ascending=False))
            else:
                st.write('Aucune valeur manquante dans les données.')

            # Remplacer les valeurs manquantes
            dataframe['Arrival Delay in Minutes'].fillna(dataframe['Arrival Delay in Minutes'].median(), inplace=True)
            st.write('Les valeurs manquantes dans la colonne "Arrival Delay in Minutes" ont été remplacées par la médiane.')

            # Afficher les données
            st.write('Les données :')
            st.write(dataframe.head())

        



            

    else: 
        st.warning("Veuillez choisir un fichier CSV")
        
        
with tab2:
    if dataframe is not None:
        st.write("# Analyse exploratoire des données")
        # Créer un graphique camembert pour la colonne "satisfaction"
        fig, ax = plt.subplots()
        ax.pie(dataframe.satisfaction.value_counts(), labels=["Neutral or dissatisfied", "Satisfied"],
        colors=sns.color_palette("YlOrBr"), autopct = '%1.1f%%')
        ax.set_title("Satisfaction")
        # Afficher le graphique camembert
        st.write(fig)
        st.write("<p style='font-weight: bold; text-align: center;'>Comme la pie chart le montre, la sélection est plus ou moins équilibrée.</p><br>", unsafe_allow_html=True)
        st.write(" ")
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
        # Sélectionner les index des variables catégorielles
        categorical_indexes = [0, 1, 3, 4] + list(range(6, 20))

        # Extraire les variables catégorielles
        categ = dataframe.iloc[:,categorical_indexes]

        # Créer une figure avec des sous-graphiques
        fig, axes = plt.subplots(6, 3, figsize = (20, 20))
        for i, col in enumerate(categ):
            column_values = dataframe[col].value_counts()
            labels = column_values.index
            sizes = column_values.values
            axes[i//3, i%3].pie(sizes, labels = labels, colors = sns.color_palette("YlOrBr"), autopct = '%1.0f%%', startangle = 90)
            axes[i//3, i%3].axis('equal')
            axes[i//3, i%3].set_title(col)

        # Ajuster l'espacement entre les sous-graphiques
        plt.subplots_adjust(wspace=0.5, hspace=0.5)

        # Afficher le graphique
        st.pyplot(fig)
        num_vars = ['Age', 'Flight Distance']

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        axs = axs.flatten()

        for i, var in enumerate(num_vars):
            sns.histplot(x=var, data=dataframe, ax=axs[i])

        fig.tight_layout()

        # Afficher le graphique avec Streamlit
        st.pyplot(fig)
        # Créer une figure avec une taille de 8x8 et une résolution de 100 dpi
        plt.figure(figsize=(8,8),dpi=100)

        # Tracer un graphique de dispersion
        sns.scatterplot(x="Departure Delay in Minutes", y="Arrival Delay in Minutes", hue="satisfaction", data=dataframe, edgecolor="black")

        # Afficher le graphique
        st.pyplot(plt)
        # Affichage de la heatmap
        st.write("## Heatmap des corrélations")
        plt.figure(figsize=(15,12))
        sns.heatmap(dataframe.corr(), annot=False, fmt='.2g')
        st.pyplot()

with tab3:
    if dataframe is not None:
        import streamlit as st
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score


        # Sélectionner les variables d'entrée et la variable cible
        X = data[['Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class', 'Flight Distance',
          'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
          'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
          'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling',
          'Checkin service', 'Inflight service', 'Cleanliness', 'Departure Delay in Minutes',
          'Arrival Delay in Minutes']]
        y = data['satisfaction']
    
        # Convertir les variables catégorielles en variables binaires
        X = pd.get_dummies(X)

        # Diviser les données en ensembles d'entraînement et de test (Test size 20% and Train size 80%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Créer un objet RandomForestClassifier
        rfc = RandomForestClassifier(n_estimators=200, max_depth=None, max_features='sqrt')

        # Entraîner le modèle avec les données d'entraînement
        rfc.fit(X_train, y_train)

        # Faire des prédictions sur les données de test
        y_pred = rfc.predict(X_test)

        # Calculer l'exactitude du modèle
        accuracy = accuracy_score(y_test, y_pred)

        # Afficher l'exactitude du modèle
        st.write('L\'exactitude du modèle est de :', accuracy)

        
        
        
with tab4:
    if dataframe is not None:
        
        
        
        
with tab5:
    if dataframe is not None:  
        
        
with tab6:
    if dataframe is not None:






        

        
