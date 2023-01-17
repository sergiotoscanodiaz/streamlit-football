from PIL import Image
import streamlit as st
import pandas as pd
import joblib

image = Image.open('football.jpg')
st.image(image, width = 600)

st.header("Pronóstico de goles")
st.text("A Streamlit Machine Learning App \nby Sergio Toscano Díaz - Máster FP IA y BD")

# Add a text input
col1, col2 = st.columns(2)
with col1:
    shots = st.number_input("Tiros por partido:", min_value=7.0, max_value=20.0)
    possession = st.number_input("Porcentaje de posesión:", min_value=30.0, max_value=100.0)
with col2:
    aerials = st.number_input("Duelos aéreos ganados por partido:", min_value=9.0, max_value=27.0)
    passes = st.number_input("Porcentaje de pases:", min_value=50.00, max_value=100.0)
rating = st.number_input("Calificación general del equipo:", min_value=6.0, max_value=7.0)
  

# Display the entered name
if st.button("Submit"):
    football_model = joblib.load("football_model.pkl")    
    X = pd.DataFrame([[shots, possession, passes, aerials, rating]],
                     columns = ["Shots pg", "Possession%", "Pass%", "AerialsWon", "Rating"])
    prediction = football_model.predict(X)[0]
    st.text(f"Este equipo se estima que marcará {prediction} goles este año")