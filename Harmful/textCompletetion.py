import json
import os
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"

from transformers import TextStreamer

import sys

sys.path.append("./Utility")

from configuation import getConfig
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer



config_path = "./config.json"
_, MODEL_NAME, _, PEFT_MODEL, max_seq_length = getConfig(config_path)


config = PeftConfig.from_pretrained(PEFT_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, PEFT_MODEL)


input_csv_path = "./data/prompts.csv"  
output_csv_path = "./data/output_file.csv"  
df = pd.read_csv(input_csv_path)

prompts = df['Behavior'].tolist()


answers = []

mistral_prompt = """### Conversation:
    
User: {}
    
Assistant: {}"""


for prompt in prompts:
    inputs = tokenizer(
        [mistral_prompt.format(prompt, "")], 
        return_tensors="pt"
    ).to("cuda")


    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        pad_token_id=tokenizer.eos_token_id  #  pad_token_id
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answers.append(answer)
    print(answer)

df['answer'] = answers


df.to_csv(output_csv_path, index=False)

print(f"Output saved to {output_csv_path}")