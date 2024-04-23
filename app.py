import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
from io import BytesIO

def extract_text_from_pdf(file):
    # Load PDF file from uploaded file
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

def parse_text_to_df(text):
    # Assuming data follows a specific format with 'Polisnummer', 'Debet', and 'Credit'
    lines = text.split('\n')
    data = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 3 and parts[0].isdigit():  # Simple validation assuming 'Polisnummer' is numeric
            try:
                polisnummer = parts[0]
                debet = float(parts[1].replace('.', '').replace(',', '.').replace('€', ''))
                credit = float(parts[2].replace('.', '').replace(',', '.').replace('€', ''))
                data.append([polisnummer, debet, credit])
            except ValueError:
                continue  # Skip lines that do not conform to expected format
    return pd.DataFrame(data, columns=['Polisnummer', 'Debet', 'Credit'])

def find_discrepancies(df1, df2):
    # Merge dataframes on 'Polisnummer' and calculate differences
    merged = pd.merge(df1, df2, on='Polisnummer', suffixes=('_1', '_2'))
    merged['Debet_diff'] = merged['Debet_1'] - merged['Debet_2']
    merged['Credit_diff'] = merged['Credit_1'] - merged['Credit_2']
    return merged[(merged['Debet_diff'] != 0) | (merged['Credit_diff'] != 0)]

# Streamlit UI setup
st.title('Discrepancy Checker for Rekening Couranten')
uploaded_file1 = st.file_uploader("Upload your company's file", type=['pdf'], key='file1')
uploaded_file2 = st.file_uploader("Upload Felison's file", type=['pdf'], key='file2')

if st.button('Analyze Discrepancies'):
    if uploaded_file1 is not None and uploaded_file2 is not None:
        text1 = extract_text_from_pdf(uploaded_file1)
        text2 = extract_text_from_pdf(uploaded_file2)
        df1 = parse_text_to_df(text1)
        df2 = parse_text_to_df(text2)
        discrepancies = find_discrepancies(df1, df2)
        if not discrepancies.empty:
            st.write('Discrepancies found:')
            st.dataframe(discrepancies)
        else:
            st.success('No discrepancies found!')
    else:
        st.error("Please upload both files to proceed.")
