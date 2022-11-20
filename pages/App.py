""" This is a entrypoint on Streamlit Cloud
    for the OCR Comparator app, histed on Hugging Face
"""
import streamlit as st

st.set_page_config(page_title="OCR Comparator", layout="wide")

st.title("OCR solutions comparator")
st.markdown("##### *EasyOCR, PPOCR, MMOCR, Tesseract*")


st.write("")
st.write("#### Here the app, hosted on Hugging Face :")

st.write("")
st.image('hg_space.PNG')
st.write("")
st.write("#####   👉  \
         [Streamlit OCR Comparator](https://huggingface.co/spaces/Loren/Streamlit_OCR_comparator)")
