import streamlit as st

st.set_page_config(page_title='OCR Comparator', layout ="wide")
st.image('ocr.png')
st.markdown(
    """
        <style>
            .sidebar .sidebar-content {{
                width: 375px;
            }}
        </style>
    """,
    unsafe_allow_html=True
)
st.write("")

st.markdown('''#### OCR, or Optical Character Recognition, is a computer vision task, \
which includes the detection of text areas, and the recognition of characters.''')
st.write("")
st.write("")

st.markdown("#####  This app allows you to compare, from a given image, the results of different solutions:")
st.markdown("##### *EasyOcr, PaddleOCR, MMOCR, Tesseract*")
st.write("")
st.write("")
st.markdown("👈 Select the **About** page from the sidebar for information on how the app works")

st.markdown("👈 or directly select the **App** page")