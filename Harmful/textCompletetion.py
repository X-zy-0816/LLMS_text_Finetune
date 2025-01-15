import json
import os

config_path = "././config.json"

from Utility.configuation import getConfig
HF_TOKEN, MODEL_NAME, DATAPATH, PEFTMODEL, max_seq_length = getConfig(config_path)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig


base_model_name = MODEL_NAME 
peft_model_dir = PEFTMODEL

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

peft_model = PeftModel.from_pretrained(base_model, peft_model_dir)
peft_model.eval() 

generator = pipeline("text-generation", model=peft_model, tokenizer=tokenizer, device=0)

input_file = "prompts.csv"  
output_file = "answers.csv" 
data = pd.read_csv(input_file)


def generate_answer(prompt):
    response = generator(prompt, max_length=100, num_return_sequences=1)
    return response[0]['generated_text']


data['Answer'] = data['Prompt'].apply(generate_answer)


data.to_csv(output_file, index=False)
print(f" {output_file}")