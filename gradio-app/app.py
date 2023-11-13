import os
import gradio as gr
import shutil
import time
import warnings

warnings.filterwarnings("ignore")
import textwrap
import langchain
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
### Multi-document retriever
from langchain.vectorstores import Chroma, FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from InstructorEmbedding import INSTRUCTOR

import glob
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import uuid
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from chromadb.config import Settings
from langchain.prompts import PromptTemplate


tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-alpha")

llm_model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-alpha", #meta-llama/Llama-2-13b-chat-hf
                                                     load_in_4bit=True,
                                                     device_map='auto',
                                                     torch_dtype=torch.float16,
                                                     low_cpu_mem_usage=True
                                                    )
max_len = 4096
llm_task = "text-generation"
#llm_task = "text-generation"
T = 0.1

llm_pipeline = pipeline(
    task=llm_task,
    model=llm_model, 
    tokenizer=tokenizer, 
    max_length=max_len,
    temperature=T,
    top_p=0.95,
    repetition_penalty=1.15
)

text_llm = HuggingFacePipeline(pipeline=llm_pipeline)

# Prompt Template for Langchain
template = """You are a helpful AI assistant, answer the below question in detail.
Question:{question}
>>Answer<<"""
prompt_template = PromptTemplate(input_variables=["question"], template = template)


text_chain = LLMChain(llm=text_llm, prompt=prompt_template)

def api_wrapper(question):
    sum_text = text_chain(question)
    
    return sum_text["text"]