import streamlit as st
import datetime


st.title("Iris flower Detection")
st.header("Data Preprocessing and Model Training")
st.subheader("Random Forest Model")
st.text("This is a simple iris flower detection app using Streamlit.")
st.write("The app allows users to input features of an iris flower and predicts its species using a pre-trained Random Forest model.")


color = st.color_picker("Pick A Color", "#ffffff")
st.write("The current color is", color)

age=st.slider("How old are you?", 0, 100, 18)
st.write("I'm ", age, 'years old')

genre = st.radio(
    "What's your favorite movie genre",
    [":rainbow[Comedy]", "***Drama***", "Documentary :movie_camera:"],
    captions=[
        "Laugh out loud.",
        "Get the popcorn.",
        "Never stop learning.",
    ],
)

st.write("You selected:", genre)

d = st.date_input("When's your birthday", datetime.date(2007, 3, 8 ))
st.write("Your birthday is:", d)


import pandas as pd 
from numpy.random import default_rng as rng

df = pd.DataFrame(rng(0).standard_normal((20, 4)), columns=["a", "b", "c","d"])

st.bar_chart(df)