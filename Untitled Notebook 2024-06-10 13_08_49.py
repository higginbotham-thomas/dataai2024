# Databricks notebook source
# MAGIC %pip install torch
# MAGIC %pip install transformers
# MAGIC

# COMMAND ----------

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Function to generate a response from the LLM
def query_llm(prompt):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Example query
prompt = "What is the capital of France?"
answer = query_llm(prompt)
print(f"Question: {prompt}\nAnswer: {answer}")

