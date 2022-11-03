import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st

df = pd.read_csv("knn/IRIS.csv")

X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6752)

knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train, y_train)

st.title("Quelle magnifique iris !")

sepal_length = st.slider("sepal_length : ", min_value=4., max_value=8., step=0.1)
sepal_width = st.slider("sepal_width : ", min_value=2., max_value=5., step=0.1)
petal_length = st.slider("petal_length : ", min_value=0., max_value=8., step=0.1)
petal_width = st.slider("petal_width : ", min_value=0., max_value=3., step=0.1)

elt_en_cours = {
    "sepal_length" : [sepal_length],
    "sepal_width": [sepal_width],
    "petal_length": [petal_length],
    "petal_width": [petal_width],
    "species": "Votre iris"
}

elt_en_cours = pd.DataFrame(elt_en_cours)
df_a_afficher = pd.concat([df, elt_en_cours])

if st.button("Calculer"):
    st.success(knn.predict([[sepal_length, sepal_width, petal_length, petal_width]]))
    col1, col2 = st.columns(2)
    col1.pyplot(sns.FacetGrid(df_a_afficher, hue="species", height=6).map(plt.scatter, "petal_width", "petal_length").add_legend())
    col2.pyplot(sns.FacetGrid(df_a_afficher, hue="species", height=6).map(plt.scatter, "sepal_width", "sepal_length").add_legend())