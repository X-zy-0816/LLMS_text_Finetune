"""
    Function: This script is used to complete the text of the prompts(harmful behaviors)


"""

input_csv_path = "./data/prompts_cn.csv"
output_csv_path = "./data/harmful/harmful_prompts_cn.csv"

import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel
import sys
sys.path.append("./Utility")

from configuation import getConfig

# Set up environment
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
config_path = "./config.json"
_, _, _, PEFT_MODEL, _, _ = getConfig(config_path)

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
df = pd.read_csv(input_csv_path)

# Ensure the output column exists
if "answer" not in df.columns:
    df["answer"] = ""

# Define prompt template
# Define prompt template
mistral_prompt = """
<|im_start|> system: "请使用中文回答下面提出的所有问题和要求 <|im_end|>\n
<|im_start|> User: {} <|im_end|>\n
<|im_start|> Assistant: {}
"""

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
    df.to_csv(output_csv_path, index=False)

print(f"Updated prompts saved to {output_csv_path}")
