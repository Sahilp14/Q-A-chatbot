import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = 'true'
os.environ["LANGCHAIN_PROJECT"] = "Q&A ChatBot with OPENAI"


# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that helps people find information."),
        ("user","Question: {question}" ),
    ]
)

def generate_response(question, api_key, llm, temprature, max_tokens):
    openai.api_key = api_key
    llm = ChatOpenAI(model=llm)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question": question})
    return answer

## Title of the App
st.title("Q&A ChatBot with OPENAI")

# sidebar for API Key input
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your OPENAI API Key", type="password")

# dropdown for model selection
llm = st.sidebar.selectbox(
    "Select OPENAI Language Model",
    ["gpt-5", "gpt-4.1","gpt-4.1-mini","gpt-4.1-nano"]
) 

# Adjust response parameter

temprature = st.sidebar.slider("Temperature",min_value=0.0, max_value=1.0, value=0.6)
max_tokens = st.sidebar.slider("Max Tokens",min_value=50, max_value=300, value=150)

# main interface for user input
st.write("Go ahead and ask any question!")
user_input = st.text_area("Your Question:")

if user_input:
    response = generate_response(user_input, api_key, llm, temprature, max_tokens)
    st.write(response)
else:
    st.write("Please enter a question to get started.")



