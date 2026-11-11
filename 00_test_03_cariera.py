import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

tabel = pd.read_csv('data/Cariera.csv')

tabel.columns = tabel.columns.str.strip()
X = tabel[['Logica','Comunicare','Creativitate','Detalii']]
Y = tabel['Domeniu']

model = DecisionTreeClassifier(max_depth=4)
model.fit(X,Y)

st.title("Ghid AI de Orientare in Cariera")
st.write('Evalueaza-ti abilitatile de la 1 la 10, iar modelul iti va recomanda domeniul potrivit.')

logica = st.slider("Cat de mult iti plac matematica, logica si analiza datelor?", 1, 10, 5)
comunicare = st.slider("Cat de mult iti place sa lucrezi si sa vorbesti cu oamenii?", 1, 10, 5)
creativitate = st.slider("Cat de mult iti place sa desenezi sa scrii sau sa vi cu idei noi?", 1, 10, 5)
detalii = st.slider("Cata rabdare ai pentru detalii mici si lucruri repetitive?", 1, 10, 5)

if st.button("Descopera cariera potrivita!"):
    date_utilizator = pd.DataFrame({
        'Logica': [logica],
        'Comunicare': [comunicare],
        'Creativitate':[creativitate],
        'Detalii':[detalii]
    })

    recomandare=model.predict(date_utilizator)

    st.success(f"Domeniul recomandat pentru tine este: **{recomandare[0]}**")
