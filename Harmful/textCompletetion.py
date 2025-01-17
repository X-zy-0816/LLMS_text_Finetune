"""
    Function: This script is used to complete the text of the prompts(harmful behaviors)


"""


import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel
import sys
sys.path.append("./Utility")

from configuation import getConfig

# Set up environment
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
config_path = "./config.json"
_, MODEL_NAME, _, PEFT_MODEL, max_seq_length = getConfig(config_path)

# Load model and tokenizer
config = PeftConfig.from_pretrained(PEFT_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, PEFT_MODEL)

# Load CSV file
csv_path = "./data/prompts_cn.csv"
df = pd.read_csv(csv_path)

# Ensure the output column exists
if "answer" not in df.columns:
    df["answer"] = ""

# Define prompt template
mistral_prompt = """### Conversation:
    
User: {}
    
Assistant: {}"""

# Iterate through prompts and perform inference
for index, row in df.iterrows():
    if pd.notna(row["answer"]) and row["answer"].strip():
        continue  # Skip if an answer already exists

    prompt = row["Behavior"]
    inputs = tokenizer(
        [mistral_prompt.format(prompt, "")],
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Assistant:" in decoded_output:
        answer = decoded_output.split("Assistant:")[1].strip()
    else:
        answer = decoded_output.strip()

    # Save the result in the DataFrame
    df.at[index, "answer"] = answer
    print(f"Processed prompt: {prompt}\nAnswer: {answer}\n")

    # Save the DataFrame back to the same CSV file after each inference
    df.to_csv(csv_path, index=False)

print(f"Updated prompts saved to {csv_path}")
