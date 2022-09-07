import streamlit as st

st.title("# OCR solutions comparator")

st.write("")
st.write("")
st.write("")

st.markdown("#####  This app allows you to compare, from a given image, the results of different solutions:")
st.markdown("##### *Easy Ocr, Paddle OCR, MMOCR, Tesseract*")
st.write("")
st.write("")

st.markdown(''' The 1st step is to choose the language for the text recognition (not all solutions \
support the same languages), and then upload the image to consider. It is then possible to change \
the default values for the text area detection process, before launching the detection task \
for each solution.''')
st.write("")
st.write("")

st.markdown(''' The different results are then presented. The 2nd step is to choose one of these \
results, in order to carry out the text recognition process there. It is also possible to change \
the default settings for each solution.''')
st.write("")
st.write("")

st.markdown("###### The results appear in 2 formats:")
st.markdown(''' - a visual format resumes the initial image, replacing the detected areas with \
the recognized text. The background is + or - strongly colored in green according to the \
confidence level of the recognition.  
    A slider allows you to change the font size, another \
allows you to modify the confidence threshold above which the text color changes: if it is at \
70% for example, then all the texts with a confidence threshold higher or equal to 70 will appear \
in white, in black otherwise.''')
st.write("")
st.markdown(" - a detailed format presents the results in a table, for each text box detected.")