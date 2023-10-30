import os
import gradio
import shutil
import random
import time
import warnings
warnings.filterwarnings("ignore")
import textwrap
import langchain
from langchain.llms import HuggingFacePipeline
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline


from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
langchain.verbose = True

def main():

    # Configure gradio QA app 
    print("Configuring gradio app")
    gradio.Markdown("Coding Assistant")
    demo = gradio.Interface(fn=get_responses, 
                            inputs=gradio.Textbox(label="Question", placeholder=""),
                            outputs=[
                                     gradio.Textbox(label="Generated Code")],
                            allow_flagging="never")
    # Launch gradio app
    print("Launching gradio app")
    demo.launch(share=True,
                enable_queue=True,
                show_error=True,
                server_name='127.0.0.1',
                server_port=int(os.getenv('CDSW_APP_PORT')))
    print("Gradio app ready")
class CFG:
    model_name = 'code-llama' # wizardlm, llama-2, bloom, falcon-40

##### Uncomment the below lines for Llama-2 #####
#access_token = os.environ["HF_TOKEN"]
#!huggingface-cli login --token $HF_TOKEN
    
def get_model(model = CFG.model_name):
    """
    Given a model name, downloads the appropriate tokenizer and pre-trained model.
    
    Args:
    - model (str): Model name, defaults to CFG.model_name
    
    Returns:
    - tokenizer: Tokenizer instance for the given model.
    - model: Pre-trained model instance.
    - max_len (int): Maximum sequence length for the given model.
    - task (str): Task the model is suited for.
    - T (float): Temperature value for the model's text-generation.
    """
    print('\nDownloading model: ', model, '\n\n')
    
    if CFG.model_name == 'falcon-40':
        tokenizer = AutoTokenizer.from_pretrained('tiiuae/falcon-40b-instruct')
        
        model = AutoModelForCausalLM.from_pretrained('tiiuae/falcon-40b-instruct',
                                                     load_in_8bit=True,
                                                     device_map='auto',
                                                     torch_dtype=torch.float16,
                                                     low_cpu_mem_usage=True,
                                                     trust_remote_code=True
                                                    )
        max_len = 4096
        task = "text-generation"
        T = 0
        
    elif CFG.model_name == 'llama-2':
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf") #meta-llama/Llama-2-13b-chat-hf
        
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", #meta-llama/Llama-2-13b-chat-hf
                                                     load_in_bit=True,
                                                     device_map='auto',
                                                     torch_dtype=torch.float16,
                                                     low_cpu_mem_usage=True,
                                                     #token=access_token
                                                    )
        max_len = 4096
        task = "text-generation"
        T = 0.1

    elif CFG.model_name == 'code-llama':
        tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-13b-Instruct-hf")
        
        model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-13b-Instruct-hf",
                                                     load_in_4bit=True,
                                                     device_map='auto',
                                                     torch_dtype=torch.float16,
                                                     low_cpu_mem_usage=True,
                                                     use_safetensors=False
                                                    )
        max_len = 2048
        task = "text-generation"
        T = 0
        
    elif CFG.model_name == 'falcon-7':
        tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
        
        model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct",
                                                     #load_in_8bit=True,
                                                     device_map='auto',
                                                     torch_dtype=torch.float16,
                                                     #low_cpu_mem_usage=True,
                                                     trust_remote_code=True
                                                    )
        max_len = 2048
        task = "text-generation"
        T = 0        
        
    else:
        print("Not implemented model (tokenizer and backbone)")
        
    return tokenizer, model, max_len, task, T

tokenizer, model, max_len, task, T = get_model(CFG.model_name)
  #Builds Transformers Pipeline to be used for inference
pipe = pipeline(
    task=task,
    model=model, 
    tokenizer=tokenizer, 
    max_length=max_len,
   temperature=T,
   top_p=0.95,
   repetition_penalty=1.15
)

llm = HuggingFacePipeline(pipeline=pipe)

def get_responses(question):
    # Prompt Template for Langchain
    template = """You are a helpful, respectful and honest assistant with a deep knowledge of code and software design. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
    Question:{context}
    """

    prompt_template = PromptTemplate(
    input_variables=["context"],
    template=template,
    )
    #final_prompt = prompt_template.format(context=question)
    text_chain = LLMChain(llm=llm, prompt=prompt_template)
    code_answer = text_chain(question)
    return code_answer["text"]

if __name__ == "__main__":
    main()