import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import torch

compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

model_name = "codellama/CodeLlama-7b-Python-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_auth_token="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")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token="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"
)

def opt_args_value(args, arg_name, default):
  if arg_name in args.keys():
    return args[arg_name]
  else:
    return default

def generate(prompt, max_new_tokens=50, temperature=70, repetition_penalty=1.0, num_beams=1, top_p=1.0, top_k=0):
  batch = tokenizer(prompt, return_tensors='pt').to('cuda')
  with torch.cuda.amp.autocast():
    output_tokens = model.generate(**batch,
                                    max_new_tokens=max_new_tokens,
                                    repetition_penalty=repetition_penalty,
                                    temperature=temperature,
                                    num_beams=num_beams,
                                    top_p=top_p,
                                    top_k=top_k)
  #prompt_length = len(prompt)
  #return tokenizer.decode(output_tokens[0], skip_special_tokens=True)[prompt_length:]
  print(temperature)
  return tokenizer.decode(output_tokens[0], skip_special_tokens=True)

def api_wrapper(args):
  # Pick up args from model api
  prompt = args["prompt"]
  
  # Pick up or set defaults for inference options
  # TODO: Set min and max for each of these
  # TODO: More intelligent control of max_new_tokens
  temperature = float(opt_args_value(args, "temperature", 70))
  max_new_tokens = float(opt_args_value(args, "max_new_tokens", 50))
  top_p = float(opt_args_value(args, "top_p", 1.0))
  top_k = int(opt_args_value(args, "top_k", 0))
  repetition_penalty = float(opt_args_value(args, "repetition_penalty", 1.0))
  num_beams = int(opt_args_value(args, "num_beams", 1))
  
  return generate(prompt, max_new_tokens, temperature, repetition_penalty, num_beams, top_p, top_k)
