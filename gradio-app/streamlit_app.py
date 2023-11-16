import plotly.express as px
import streamlit as st
import pandas as pd
from huggingface_hub import InferenceClient
import os
import requests
import json

st.set_page_config(layout="wide")

st.title('CML LLM Playground 🔬🧪')

llm_tab, vectordb_tab, llmops_tab = st.tabs(["LLM", "VectorDB", "LLMOps"])


columns = ['Time', 'LLM_Model', 'VectorDB', 'Latency(s)', 'Chunk_Size', 'Chunk_Overlap', 'Prompt', 'Question', 'Answer']
logging_df = pd.DataFrame(columns=columns)
#API_KEY = os.environ["API_KEY"]
#client = InferenceClient(model="meta-llama/Llama-2-70b-chat-hf", token=API_KEY)
DOMAIN = os.environ["CDSW_DOMAIN"]
API_URL = f'https://modelservice.{DOMAIN}/model'
ACCESS_KEY = os.environ["CDSW_API_KEY"]
HEADERS = {'Content-Type': 'application/json'}
def generate_response(api_url, cml_access_key, headers, question):
    api_url = api_url
    cml_access_key = cml_access_key
    question = {'question': question}
    data = json.dumps({'accessKey': cml_access_key, 'request': question})
    headers = headers
    response = requests.post(api_url, data = data, headers = {'Content-Type': 'application/json'})
    try:
        response_json = response.json()
        return response_json.get('response')
    except json.JSONDecodeError:
        return "Failed to parse JSON response"  



with llm_tab:
    llm_tab_col1, llm_tab_col2 = st.columns(2)
    with llm_tab_col1:
        option = st.selectbox(
        'Add Model Name',
        ('Llama-2-7B', 'Zephyr-7B', 'Mistral-7B'))
        model_endpoint = st.text_input('Set CML Model Endpoint')
        input_prompt = st.text_input('Input Prompt')
        st.write('The current selected LLM is: ', option)
    with llm_tab_col2:
        question = st.text_input('Question')
        submit = st.button("Submit")
        if submit:
            output = generate_response(API_URL, ACCESS_KEY, HEADERS, question)
            output_textbox = st.text_area("Generated Text", output)
        


with vectordb_tab:
    vectordb_tab_col1, vectordb_tab_col2 = st.columns(2)
    with vectordb_tab_col1:
        vec_ip = st.text_input("VectorDB IP")
        vec_port = st.text_input("VectorDB Port")
    with vectordb_tab_col2:
        st.dataframe(logging_df)
        
        
with llmops_tab:
    llmops_col1, llmops_col2 = st.columns(2)
    with llmops_col1:
        st.dataframe(logging_df)


    with llmops_col2:
        df = px.data.gapminder()
        fig = px.scatter(
        df.query("year==2007"),
        x="gdpPercap",
        y="lifeExp",
        size="pop",
        color="continent",
        hover_name="country",
        log_x=True,
        size_max=60,
        )
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)