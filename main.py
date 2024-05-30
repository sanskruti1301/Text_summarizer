import streamlit as st
from io import StringIO
from pdf_txt import pdf_to_text
from summarizer1 import summarizer
from pathlib import Path
import os

st.markdown("<h1 style='text-align: center;'>Text Summarizer</h1>",
unsafe_allow_html=True)


file = st.file_uploader("Please choose a file", type=['txt', 'pdf'])
st.markdown("<h5 style='text-align: center;'>OR</h5>", unsafe_allow_html=True)
text = st.text_area("Input Text For Summary (More than 200 words)", height=200)
col1,  col2, col3 = st.columns(3)
if col1.button('SUMMARIZE'):

        if file is not None:
            if bool(text)== True:
                st.error("ERROR: YOU CAN'T ENTER BOTH")
                st.stop()
            else:
                if file.name[-3:] == "pdf":
                    path=Path(file.name)
                    path.write_bytes(file.getvalue())
                    text = pdf_to_text(file.name)

                else:
                    stringio = StringIO(file.getvalue().decode("utf-8"))
                    text=stringio.read()
    
        textfile = open("content.txt","w")
        textfile.write(text)
        textfile.close()
        summary1=summarizer(text)

        st.markdown("<h2> Spacy Summary  </h2>" ,  unsafe_allow_html=True)        
        st.markdown("<h3> The summarized text : </h3>" ,  unsafe_allow_html=True)                                
        st.write(summary1)
       