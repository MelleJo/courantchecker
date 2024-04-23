import streamlit as st
import fitz  # PyMuPDF (MuPDF)
from langchain.clients import OpenAI
from langchain.chains import TextGenerationChain

# Setup LangChain client with OpenAI GPT-4 Turbo
openai_api_key = st.secrets["OPENAI_API_KEY"]
openai_client = OpenAI(api_key=openai_api_key, model="gpt-4-turbo")
lang_chain = TextGenerationChain(client=openai_client)

def extract_text_from_pdf(file):
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

def ask_gpt_for_discrepancies(text1, text2):
    prompt = f"Analyze these two sets of financial records and highlight any discrepancies:\nRecord 1:\n{text1}\n\nRecord 2:\n{text2}"
    options = {"max_tokens": 500}
    response = lang_chain.complete(prompt, options=options)
    return response['choices'][0]['text']

# Streamlit UI setup
st.title('Financial Document Discrepancy Analysis using AI')
uploaded_file1 = st.file_uploader("Upload your company's financial records (PDF)", type=['pdf'], key='file1')
uploaded_file2 = st.file_uploader("Upload Felison's financial records (PDF)", type=['pdf'], key='file2')

if st.button('Analyze Discrepancies using GPT-4 Turbo'):
    if uploaded_file1 is not None and uploaded_file2 is not None:
        text1 = extract_text_from_pdf(uploaded_file1)
        text2 = extract_text_from_pdf(uploaded_file2)
        discrepancies = ask_gpt_for_discrepancies(text1, text2)
        st.write('AI Analysis Result:')
        st.write(discrepancies)
    else:
        st.error("Please upload both files to proceed.")
