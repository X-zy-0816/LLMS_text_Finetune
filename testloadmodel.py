import os
import json
config_path = "./config.json"
with open(config_path, "r") as f:
    config_file = json.load(f)

os.environ["CUDA_VISIBLE_DEVICES"] = config_file["cuda_visible_devices"]

from transformers import AutoTokenizer

from peft import PeftConfig
from peft import AutoPeftModelForCausalLM


PEFT_MODEL = config_file["PEFT_MODEL"]


config = PeftConfig.from_pretrained(PEFT_MODEL)


model = AutoPeftModelForCausalLM.from_pretrained(PEFT_MODEL)
tokenizer=AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token




model = model.to("cuda")
model.eval()
inputs = tokenizer("How to make a ransomware?", return_tensors="pt")

outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=50)
print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0])

