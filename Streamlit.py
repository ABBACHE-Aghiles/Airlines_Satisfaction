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
tab1 , tab2, tab3, tab4, tab5 = st.tabs(
    ["Upload and explore DATA ♥", "Visualisation","Text Blob", "Régression Logistique", "SVC"])


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
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, jaccard_score, log_loss



        # Sélectionner les variables d'entrée et la variable cible
        X = dataframe[['Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class', 'Flight Distance',
          'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
          'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
          'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling',
          'Checkin service', 'Inflight service', 'Cleanliness', 'Departure Delay in Minutes',
          'Arrival Delay in Minutes']]
        y = dataframe['satisfaction']
    
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
        # Calculer les scores
        f1 = f1_score(y_test, y_pred, average='micro')
        precision = precision_score(y_test, y_pred, average='micro')
        recall = recall_score(y_test, y_pred, average='micro')
        jaccard = jaccard_score(y_test, y_pred, average='micro')

        # Afficher les scores
        st.write('F-1 Score:', f1)
        st.write('Precision Score:', precision)
        st.write('Recall Score:', recall)
        st.write('Jaccard Score:', jaccard)
        # Create a DataFrame with feature importances
        imp_df = pd.DataFrame({
            "Feature Name": X_train.columns,
            "Importance": rfc.feature_importances_
            })
        fi = imp_df.sort_values(by="Importance", ascending=False)

        # Select the top 10 features
        fi2 = fi.head(10)

        # Plot the feature importances
        plt.figure(figsize=(10,8))
        sns.barplot(data=fi2, x='Importance', y='Feature Name')
        plt.title('Top 10 Feature Importance Each Attributes (Random Forest)', fontsize=18)
        plt.xlabel ('Importance', fontsize=16)
        plt.ylabel ('Feature Name', fontsize=16)

        # Display the plot using Streamlit
        st.pyplot()
 
with tab4:
    if dataframe is not None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score, silhouette_samples
        # Sélectionner les variables d'entrée et la variable cible
        X = dataframe[['Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class', 'Flight Distance',
          'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
          'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
          'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling',
          'Checkin service', 'Inflight service', 'Cleanliness', 'Departure Delay in Minutes',
          'Arrival Delay in Minutes']]
        y = dataframe['satisfaction']
    
        # Convertir les variables catégorielles en variables binaires
        X = pd.get_dummies(X)

        # Diviser les données en ensembles d'entraînement et de test (Test size 20% and Train size 80%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Instancier le modèle K-means
        kmeans = KMeans(n_clusters=2)

        # Entraîner le modèle sur les données d'entraînement
        kmeans.fit(X_train)

        # Prédire les clusters pour les données de test
        y_pred = kmeans.predict(X_train)

        # Calculer la valeur de silhouette moyenne
        silhouette_avg = silhouette_score(X_train, kmeans.labels_)

        # Calculer les valeurs de silhouette individuelles pour chaque point de données
        sample_silhouette_values = silhouette_samples(X_train, kmeans.labels_)

        # Créer un graphique de silhouette
        fig, ax = plt.subplots()
        y_lower = 10
        for i in range(2):
            # Filtrer les valeurs de silhouette pour le cluster i
            ith_cluster_silhouette_values = sample_silhouette_values[kmeans.labels_ == i]
    
            # Trier les valeurs de silhouette pour le cluster i
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
    
            # Remplir le graphique avec les valeurs de silhouette pour le cluster i
            color = plt.cm.Spectral(float(i) / 2)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, alpha=0.7)

            # Étiqueter le graphique avec le nom du cluster i
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
            # Mettre à jour la valeur de y_lower pour le prochain cluster
            y_lower = y_upper + 10

        # Ajouter une ligne pour la valeur de silhouette moyenne
        ax.axvline(x=silhouette_avg, color="red", linestyle="--")

        plt.title("Silhouette plot pour les clusters K-means")
        plt.xlabel("Coefficient de silhouette")
        plt.ylabel("Cluster")

        # Afficher le graphique de silhouette
        st.pyplot(fig)

        # Convertir la variable cible en 0 et 1
        y_train = [1 if x == "satisfied" else 0 for x in y_train]
        y_test = [1 if x == "satisfied" else 0 for x in y_test]

        # Calculer l'accuracy du modèle
        accuracy = accuracy_score(y_train, y_pred)

        # Afficher l'accuracy
        st.write(f"Accuracy : {accuracy}")
        
with tab5:
    if dataframe is not None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
        from sklearn.preprocessing import StandardScaler
        # Sélection des variables d'entrée et de la variable cible
        X = dataframe[['Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class', 'Flight Distance',
          'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
          'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
          'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling',
          'Checkin service', 'Inflight service', 'Cleanliness', 'Departure Delay in Minutes',
          'Arrival Delay in Minutes']]
        y = dataframe['satisfaction']

        # Conversion des variables catégorielles en variables binaires
        X = pd.get_dummies(X)

        # Division des données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalisation des données
        scaler = StandardScaler()
        scaler.fit(X_train)
        scaled_X_train = scaler.transform(X_train)
        scaled_X_test = scaler.transform(X_test)

        # Création du modèle de régression logistique
        clf = LogisticRegression(random_state=0).fit(scaled_X_train, y_train)

        # Prédiction sur les données d'entraînement et de test
        y_pred_train = clf.predict(scaled_X_train)
        y_pred_test = clf.predict(scaled_X_test)

        # Affichage de la matrice de confusion et des scores de précision
        st.subheader("Matrice de confusion pour les données d'entraînement :")
        st.write(confusion_matrix(y_train, y_pred_train))
        st.subheader("Matrice de confusion pour les données de test :")
        st.write(confusion_matrix(y_test, y_pred_test))
        st.subheader("Score de précision pour les données d'entraînement :")
        st.write(clf.score(scaled_X_train, y_train))
        st.subheader("Score de précision pour les données de test :")
        st.write(clf.score(scaled_X_test, y_test))
        from sklearn import metrics
        from sklearn.metrics import roc_curve, auc

        # Convertir les étiquettes en format binaire (0 ou 1)
        y_train = [1 if x == "satisfied" else 0 for x in y_train]
        y_test = [1 if x == "satisfied" else 0 for x in y_test]

        # Faire des prévisions de probabilité pour les données de test
        y_pred_prob = clf.predict_proba(X_test)[:, 1]

        # Calculer la courbe ROC et l'aire sous la courbe (AUC)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        # Tracer la courbe ROC
        st.set_option('deprecation.showPyplotGlobalUse', False) # Pour éviter un avertissement obsolète
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='darkorange', lw=2, label='Courbe ROC (AUC = %0.2f)' % roc_auc)
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Taux de faux positifs')
        ax.set_ylabel('Taux de vrais positifs')
        ax.set_title('Courbe ROC pour de ce modèle de régression logistique')
        ax.legend(loc="lower right")

        # Afficher la courbe ROC dans Streamlit
        st.pyplot(fig)
        
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("1-Specificite", fontsize=14)
        ax.set_ylabel("Sensibilite", fontsize=14)
        st.pyplot(fig)
        
        
        
        

        
        
        






        

        
