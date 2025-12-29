import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

# LangSmith (optional – unchanged)
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A ChatBot with GEMINI"

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that helps people find information."),
        ("user", "Question: {question}"),
    ]
)

def generate_response(question, api_key, llm, temperature, max_tokens):
    llm = ChatGoogleGenerativeAI(
        model=llm,
        google_api_key=api_key,
        temperature=temperature,
        max_output_tokens=max_tokens
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question": question})
    return answer

# Title
st.title("Q&A ChatBot with GEMINI")

# Sidebar
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your GEMINI API Key", type="password")

# Gemini model selection
llm = st.sidebar.selectbox(
    "Select GEMINI Model",
    [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ]
)

# Parameters
temperature = st.sidebar.slider("Temperature",min_value=0.0, max_value=1.0, value=0.6)
max_tokens = st.sidebar.slider("Max Tokens",min_value=50, max_value=300, value=150)

# User input
st.write("Go ahead and ask any question!")
user_input = st.text_area("Your Question:")

if user_input:
    response = generate_response(user_input, api_key, llm, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please enter a question to get started.")
