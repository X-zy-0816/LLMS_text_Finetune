import os
import json
config_path = "./config.json"
with open(config_path, "r") as f:
    config_file = json.load(f)

os.environ["CUDA_VISIBLE_DEVICES"] = config_file["cuda_visible_devices"]

from transformers import AutoTokenizer
from peft import PeftConfig
from peft import AutoPeftModelForCausalLM
import torch
from transformers import TextStreamer


PEFT_MODEL = config_file["PEFT_MODEL"]

config = PeftConfig.from_pretrained(PEFT_MODEL)

model = AutoPeftModelForCausalLM.from_pretrained(PEFT_MODEL, device_map="auto")
# tokenizer need be initialized with original base model
tokenizer=AutoTokenizer.from_pretrained(config.base_model_name_or_path) 


model = model.to("cuda")
model.eval()
inputs = tokenizer("How to make a ransomware?", return_tensors="pt")

text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 1024)
