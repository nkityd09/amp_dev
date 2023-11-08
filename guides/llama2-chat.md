# Shared llama2-chat model guide

## Prompting
Llama 2 Chat Prompt Structure
The Llama 2 chat model was fine-tuned for chat using a specific structure for prompts. This structure relied on four special tokens:

`<s>`: the beginning of the entire sequence.<<SYS>>\n: the beginning of the system message.

`\n<</SYS>>\n\n`: the end of the system message.

`[INST]`: the beginning of some instructions.

`[/INST]`: The end of instructions


With that in mind we would typically structure chat messages like so:
```
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always do...

If you are unsure about an answer, truthfully say "I don't know"
<</SYS>>

What are glaciers? [/INST]
```

## Inference Options
As part of the API wrapper script implemented in `launch_model_llama2-chat.py`, the API json can optionally accept inference options:
Adjust these during your inference requests to massage the responses you get.
Most likely max_new_tokens is the only thing you need to touch.
```
{
  "prompt": "hello",
  "temperature": "70",
  "max_new_tokens": "50",
  "top_p": "1.0",
  "top_k": "0",
  "repetition_penalty": "1.0",
  "num_beams": "1"
}
```
