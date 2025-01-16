import json
import os

config_path = "././config.json"

from Utility.configuation import getConfig
HF_TOKEN, MODEL_NAME, DATAPATH, PEFT_MODEL, max_seq_length = getConfig(config_path)

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
import pandas as pd


base_model_name = MODEL_NAME 
peft_model_dir = PEFT_MODEL

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

peft_model = PeftModel.from_pretrained(base_model, peft_model_dir)
peft_model.eval() 

generator = pipeline("text-generation", model=peft_model, tokenizer=tokenizer, device=0)

input_file = "./data/prompts.csv"  
output_file = "./data/answers.csv" 
data = pd.read_csv(input_file)


def generate_answer(prompt):
    response = generator(prompt, max_length=100, num_return_sequences=1)
    return response[0]['generated_text']


data['Answer'] = data['Prompt'].apply(generate_answer)


data.to_csv(output_file, index=False)
print(f" {output_file}")