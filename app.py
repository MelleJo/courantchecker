import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import END, MessageGraph

st.title("Courantchecker")
openai_api_key = st.secrets["OPENAI_API_KEY"]
