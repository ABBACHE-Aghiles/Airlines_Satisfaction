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

import streamlit as st
# Afficher le titre en haut et centré
st.markdown("<h1 style='text-align: center;'>Data Slayers ⚔️</h1>", unsafe_allow_html=True)
