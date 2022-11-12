"""A main script to run streamlit application.

"""
import streamlit as st
import pandas as pd
from src.utility.loggers import logger
from src.inference.predictor_factory import get_predictor

# st.set_page_config(layout="wide")
st.title("Text Clustering")
st.text("Models are trained on small sample of news data")

form = st.form(key="my-form")
input_data = form.text_area("Enter text for cluster identification")

clustering_method = form.radio(
    "Choose a text clustering model",
    (
        "nltk_kmeans",
        "sklearn_kmeans",
    ),
)
submit = form.form_submit_button("Submit")


if submit:
    try:
        predictor = get_predictor(clustering_method)

        output = predictor.get_model_output(input_data)
        st.write("model_selected:", clustering_method)
        st.write("model_output:", output)
    except Exception as error:
        message = "Error while creating output"
        logger.error(message, str(error))
